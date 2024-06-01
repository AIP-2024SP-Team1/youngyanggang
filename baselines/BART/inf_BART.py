import torch
from transformers import BartTokenizer, BartForConditionalGeneration

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

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Example inputs for inference
    inputs = [
        'Once upon a time, there was a little duckling who looked very different from the others.',
        'The ugly duckling was very sad because he felt like he didn\'t belong.'
    ]
    
    predictions = predict(model, tokenizer, inputs)
    for input_text, prediction in zip(inputs, predictions):
        print(f"Input: {input_text}\nPrediction: {prediction}\n")
