import pandas as pd
import numpy as np
import ast

scores = pd.read_csv('./output/merged(before).csv')

rouge_l = scores['Rouge-L']

sum = 0
for r in rouge_l: 
    r = ast.literal_eval(r)
    m = max(r)
    sum += m
print('rouge-L:', sum/len(rouge_l))