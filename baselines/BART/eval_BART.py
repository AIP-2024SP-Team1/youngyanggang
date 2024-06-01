import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, load_metric
from rouge_score import rouge_scorer
import bert_score
from bleurt import score as bleurt_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the dataset
data = pd.read_csv('/mnt/data/the-ugly-duckling-story.csv')

# Preprocess data
data = data.rename(columns={"input": "context", "output": "question"})

# Create a HuggingFace Dataset
dataset = Dataset.from_pandas(data)

# Load BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def preprocess_function(examples):
    inputs = examples['context']
    targets = examples['question']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length').input_ids
    model_inputs['labels'] = labels
    return model_inputs

# Tokenize data
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Evaluation metrics
rouge = load_metric('rouge')
bleurt = bleurt_score.BleurtScorer("PATH_TO_BLEURT_CHECKPOINT")
bertscore = load_metric('bertscore')

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE-L
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"])

    # BERTScore
    P, R, F1 = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang='en')

    # BLEURT
    bleurt_scores = bleurt.score(decoded_preds, decoded_labels)

    # Self-BLEU (using NLTK)
    self_bleu_scores = []
    for pred in decoded_preds:
        score = sentence_bleu([pred.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
        self_bleu_scores.append(score)

    metrics = {
        'rougeL': rouge_scores['rougeL'].mid.fmeasure,
        'bertscore_precision': P.mean(),
        'bertscore_recall': R.mean(),
        'bertscore_f1': F1.mean(),
        'bleurt': sum(bleurt_scores) / len(bleurt_scores),
        'self_bleu': sum(self_bleu_scores) / len(self_bleu_scores)
    }
    return metrics

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

print(results)
