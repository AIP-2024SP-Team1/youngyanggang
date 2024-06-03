import nltk
from nltk.util import ngrams

import json
import pandas as pd
from collections import Counter

nltk.download('punkt')

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


with open('./data/ftqa_wh_train.json') as f:
    data = json.load(f)

contexts = []
questions = []
scores = []

for obj in data:
    context = obj['input']
    question = obj['output'].split(', ')
    for q in question:
        score = compute_bi_gram_similarity(context, q)

        contexts.append(context)
        questions.append(q)
        scores.append(score)

data = pd.DataFrame({
    'context': contexts,
    'questions': questions,
    'bi-gram scores': scores
})

data.to_csv('./output/bi_gram.csv')
