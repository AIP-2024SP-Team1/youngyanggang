import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_metric
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from bleurt import score as bleurt_score

# Load data
def load_data(file_path, sample_size=None):
    data = pd.read_csv(file_path)
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)
    return data

# Generate questions using BART
def generate_questions(model, tokenizer, input_texts, max_length=50):
    questions = []
    for text in tqdm(input_texts, desc="Generating questions"):
        inputs = tokenizer.encode("Generate question: " + text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=max_length, num_beams=5, early_stopping=True)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions.append(question)
    return questions

# Calculate ROUGE-L
def calculate_rouge(predictions, references):
    rouge = load_metric('rouge')
    results = rouge.compute(predictions=predictions, references=references, rouge_types=['rougeL'])
    return results['rougeL'].mid.fmeasure

# Calculate BERTScore
def calculate_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    return F1.mean().item()

# Calculate Self-BLEU
def calculate_self_bleu(predictions):
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []
    for pred in tqdm(predictions, desc="Calculating Self-BLEU"):
        other_preds = [p for p in predictions if p != pred]
        bleu_score = sentence_bleu([other_preds], pred, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)
    return sum(bleu_scores) / len(bleu_scores)

# Calculate BLEURT
def calculate_bleurt(predictions, references, bleurt_checkpoint='BLEURT-20'):
    bleurt_scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
    scores = bleurt_scorer.score(references=references, candidates=predictions)
    return sum(scores) / len(scores)

# Main evaluation function
def evaluate(model, tokenizer, data):
    input_texts = data['context'].tolist()
    references = data['question'].tolist()
    
    predictions = generate_questions(model, tokenizer, input_texts)
    
    rouge_l = calculate_rouge(predictions, references)
    bertscore_f1 = calculate_bertscore(predictions, references)
    self_bleu = calculate_self_bleu(predictions)
    bleurt = calculate_bleurt(predictions, references, bleurt_checkpoint=bleurt_checkpoint)
    
    results = {
        "ROUGE-L": rouge_l,
        "BERTScore_F1": bertscore_f1,
        "Self-BLEU": self_bleu,
        "BLEURT": bleurt
    }
    
    return results

# Load model and tokenizer
model_name = 'facebook/bart-base'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load your dataset
data = load_data('./data/preprocessed_test.csv', sample_size=300) 

# Evaluate
results = evaluate(model, tokenizer, data)
print(results)