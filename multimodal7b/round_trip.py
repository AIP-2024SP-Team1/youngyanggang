import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import json
import pandas as pd

openai.api_key = os.environ.get('OPENAI_API_KEY')

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def inf_llama(prompt, input):
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

def inf_gpt(prompt, input, model):
    if model == 'gpt': model = 'gpt-3.5-turbo'
    else: model = 'gpt-4-turbo'

    response = openai.ChatCompletion.create(
        model = model,
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
    )
    response = response['choices'][0]['message']['content']

    return response


def gen_ans(context, question, model):
    prompt = "What is the answer of the Question? Don't add any words other than what I requested."
    input = f"Context: {context}\nQuestion: {question}\nAnswer:"
    if model == 'llama': response = inf_llama(prompt, input)
    else: response = inf_gpt(prompt, input, model)

    return response

def gen_q(context, answer, model):
    prompt = "Guess what the question might have been.\
              Don't add any words other than what I requested."
    input = f"Context: {context}\nAnswer: {answer}\nQuestion:"
    if model == 'llama': response = inf_llama(prompt, input)
    else: response = inf_gpt(prompt, input, model)

    return response

def round_trip_similarity(original_question, generated_question):
    return 1 if original_question.strip().lower() == generated_question.strip().lower() else 0

torch.cuda.empty_cache()

with open('./data/ftqa_wh_train.json') as f:
    data = json.load(f)

contexts = []
q_ans = []

q_llama = []
same_llama = [] 

q_gpt = []
same_gpt = []

q_gpt4 = []
same_gpt4 = []


for obj in data[:300]:
    context = obj['input']
    questions = obj['output'].split(', ')

    for question in questions:
        ans = gen_ans(context, question, 'llama')
        q = gen_q(context, ans, 'llama')
        same = round_trip_similarity(question, q)
        same_llama.append('Y') if same == 1 else same_llama.append('N')
        q_llama.append(q)

        ans = gen_ans(context, question, 'gpt')
        q = gen_q(context, ans, 'gpt')
        same = round_trip_similarity(question, q)
        same_gpt.append('Y') if same == 1 else same_gpt.append('N')
        q_gpt.append(q)

        ans = gen_ans(context, question, 'gpt4')
        q = gen_q(context, ans, 'gpt4')
        same = round_trip_similarity(question, q)
        same_gpt4.append('Y') if same == 1 else same_gpt.append('N')
        q_gpt4.append(q)

        contexts.append(context)
        q_ans.append(question)

data = pd.DataFrame({
    'context': contexts,
    'question': q_ans,
    'q_llama': q_llama,
    'q_gpt3.5': q_gpt,
    'q_gpt4': q_gpt4,
    'same_llama3': same_llama,
    'same_gpt3.5': same_gpt,
    'same_gpt4': same_gpt4
})
data.to_csv('./output/round_trip.csv')