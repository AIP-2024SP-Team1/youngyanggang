import numpy as np
import pandas as pd
import json
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import nltk
from nltk.util import ngrams

nltk.download('punkt')

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    force_download=True
)

def inference(prompt, input):
    messages = [
        {
            "role": "system", 
            "content": prompt
        },
        {
            "role": "user", 
            "content": input
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
    
    return response

def compute_bi_gram_similarity(context, question):
    context_tokens = nltk.word_tokenize(context)
    question_tokens = nltk.word_tokenize(question)
    
    context_bigrams = list(ngrams(context_tokens, 2))
    question_bigrams = list(ngrams(question_tokens, 2))
    
    context_bigrams_count = Counter(context_bigrams)
    question_bigrams_count = Counter(question_bigrams)
    
    intersection = sum((context_bigrams_count & question_bigrams_count).values())
    question_bigram_count = sum(question_bigrams_count.values())
    
    return intersection / question_bigram_count if question_bigram_count != 0 else 0


def generate_answer(question, context):
    prompt = "What is the answer of the Question? Don't add any words other than what I requested."
    input = f"Context: {context}\nQuestion: {question}\n"
    response = inference(prompt, input)

    return response

def generate_question(answer, context):
    prompt = "Guess what the question might have been.\
              Don't add any words other than what I requested."
    input = f"Context: {context}\nAnswer: {answer}\nQuestion:"
    response = inference(prompt, input)

    return response

def round_trip_similarity(original_question, generated_question):
    return 1 if original_question.strip().lower() == generated_question.strip().lower() else 0

def ranking(context, questions):
    bi_gram_scores = [compute_bi_gram_similarity(context, q) for q in questions]
    
    round_trip_scores = []
    for question in questions:
        answer = generate_answer(question, context)
        generated_question = generate_question(answer, context)
        if generated_question[-1] == '.': generated_question = generated_question[:-1] + '?'
        round_trip_scores.append(round_trip_similarity(question, generated_question))
    
    combined_scores = np.array(bi_gram_scores) + np.array(round_trip_scores)
    
    sorted_indices = np.argsort(-combined_scores)
    sorted_questions = [questions[idx] for idx in sorted_indices]

    return sorted_questions


torch.cuda.empty_cache()

with open('../../data/ftqa_wh_train.json') as f:
    data = json.load(f)

contexts = []
selected = []

for obj in data[4000:]:
    context = obj['input']
    questions = obj['output'].split(', ')

    ranked_questions = ranking(context, questions)
    print(ranked_questions)

    contexts.append(context)
    selected.append("\n".join(ranked_questions))

tmp = pd.DataFrame({
    'context': contexts,
    'selected': selected
})

tmp.to_csv('../../output/tmp.csv')

"""for obj in data:
    context = obj['input']
    questions = obj['output'].split(', ')

    ranked_questions = ranking(context, questions)
    print(ranked_questions)

    contexts.append(context)
    selected.append("\n".join(ranked_questions))

    if len(contexts) % 1000 == 0:
        data = pd.DataFrame({
            'context': contexts,
            'selected': selected
        })
        data.to_csv('./output/selected.csv')

data = pd.DataFrame({
    'context': contexts,
    'selected': selected
})

data.to_csv('./output/selected.csv')"""