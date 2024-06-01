import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_metric
from bert_score import score as bert_score


# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Generate questions using BART
def generate_questions(model, tokenizer, input_texts, max_length=50):
    questions = []
    for text in input_texts:
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

# Main evaluation function
def evaluate(model, tokenizer, data):
    input_texts = data['context'].tolist()
    references = data['question'].tolist()
    
    predictions = generate_questions(model, tokenizer, input_texts)
    
    rouge_l = calculate_rouge(predictions, references)
    bertscore_f1 = calculate_bertscore(predictions, references)
    
    results = {
        "ROUGE-L": rouge_l,
        "BERTScore_F1": bertscore_f1
    }
    
    return results

# Load model and tokenizer
model_name = 'facebook/bart-base'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load your dataset
data = load_data('../data/preprocessed_test.csv')

# Evaluate
results = evaluate(model, tokenizer, data)
print(results)
