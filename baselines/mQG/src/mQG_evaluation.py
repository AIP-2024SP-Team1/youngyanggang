import ast
import pandas as pd
from eval_utils import rougel_eval, selfbleu_eval, bertscore_eval, bleurt_eval, concat_gen, tokenize
from gen_answerability import run_generate

TYPE2CONTROL_SIGNAL = ['What', 'How', 'Who', 'Why', 'Where', 'When', 'Which']

# path for generated result
result_dir = 'output/pred_q.csv'
result_df = pd.read_csv(result_dir)
for i in TYPE2CONTROL_SIGNAL:
    result_df[i] = result_df[i].apply(lambda x: ast.literal_eval(x))
    
# path for golden data
test_dir = 'data/ftqa_wh_2_test1.csv'
gold_df = pd.read_csv(test_dir)
gold_df = gold_df.loc[:,['cor_section', 'question']]
gold_df.columns = ['context', 'question']

# merge dataframe
gb = gold_df.groupby(['context'])
result = pd.DataFrame(gb['question'].unique())

tot_df = pd.merge(result, result_df, on="context")
tot_df = tot_df.dropna()
tot_df['tot_gen'] = tot_df.apply(lambda x: concat_gen(x, 4, TYPE2CONTROL_SIGNAL), axis=1)

#rougel
rougel_eval(tot_df)

#selfbleu
selfbleu_eval(tot_df)

#bertscore
bertscore_eval(tot_df)

#bleurt
bleurt_eval(tot_df)