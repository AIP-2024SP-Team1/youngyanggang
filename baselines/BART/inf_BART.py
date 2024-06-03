import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import random
import pandas as pd

def load_model(model_name='facebook/bart-base'):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def predict(model, tokenizer, inputs, device='cuda'):
    model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for input_text in inputs:
            inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True, padding='max_length').to(device)
            outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
    
    return predictions

def get_random_samples(data, num_samples=2):
    return random.sample(data, num_samples)

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['context'].tolist()

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model()
    
     # Load data from CSV
    file_path = './data/preprocessed_test.csv' 
    data = load_data_from_csv(file_path)
    
    # Get random samples from the data
    input_texts = get_random_samples(data, num_samples=5)
    
    # Make predictions
    predictions = predict(model, tokenizer, input_texts)
    for input_text, prediction in zip(inputs, predictions):
        print(f"Input: {input_text}\nPrediction: {prediction}\n")
