import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from rouge_score import rouge_scorer
import bert_score
from bleurt import score as bleurt_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# Load BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Load datasets
def load_data(file_path):
    return pd.read_csv(file_path)

train_data = load_data('./data/preprocessed_train.csv')
val_data = load_data('./data/preprocessed_val.csv')
test_data = load_data('./data/preprocessed_test.csv')

# Convert datasets to Hugging Face datasets
def convert_to_hf_dataset(data):
    return {
        'context': data['context'].tolist(),
        'question': data['question'].tolist()
    }

train_dataset = convert_to_hf_dataset(train_data)
val_dataset = convert_to_hf_dataset(val_data)
test_dataset = convert_to_hf_dataset(test_data)

# Preprocess function
def preprocess_function(examples):
    inputs = examples['context']
    targets = examples['question']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length').input_ids
    model_inputs['labels'] = labels
    return model_inputs

# Tokenize datasets
tokenized_train = preprocess_function(train_dataset)
tokenized_val = preprocess_function(val_dataset)
tokenized_test = preprocess_function(test_dataset)

# Convert to dataset objects
import datasets
train_dataset = datasets.Dataset.from_dict(tokenized_train)
val_dataset = datasets.Dataset.from_dict(tokenized_val)
test_dataset = datasets.Dataset.from_dict(tokenized_test)

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
    evaluation_strategy="epoch"
)

# Evaluation metrics
rouge = load_metric('rouge')
bleurt = bleurt_score.BleurtScorer("BLEURT-20/BLEURT-20")
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
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate the model on validation set
val_results = trainer.evaluate()

print("Validation Results:")
print(val_results)

# Evaluate the model on test set
test_results = trainer.evaluate(eval_dataset=test_dataset)

print("Test Results:")
print(test_results)