import nltk
from nltk.util import ngrams

from collections import Counter

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