import pandas as pd
import numpy as np
import ast

scores = pd.read_csv('./output/merged(before).csv')

rouge_l = scores['Rouge-L']
bert_scores = scores['BERTScore']
bleurt = scores['BLEURT']
self_bleu = scores['Self-BLEU']

sum = 0
for r in rouge_l: 
    r = ast.literal_eval(r)
    m = max(r)
    sum += m
print('rouge-L:', sum/len(rouge_l))

sum = 0
for b in bert_scores:
    b = ast.literal_eval(b)
    m = max(b)
    sum += m
print('BERTScore:', sum/len(rouge_l))

sum = 0
for b in bleurt:
    b = ast.literal_eval(b)
    m = max(b)
    sum += m
print('BLEURT:', sum/len(rouge_l))

s = np.mean(list(np.mean(ast.literal_eval(item)) for item in self_bleu))
print('Self-BLEU:', s)