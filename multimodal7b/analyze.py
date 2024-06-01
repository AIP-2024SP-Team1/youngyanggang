import pandas as pd

# 데이터 로드
df = pd.read_csv('./output/merged.csv')

# 점수 컬럼과 해당하는 최소 10개 점수 추출 함수
def get_lowest(column_name):
    all_scores = []
    if column_name == 'Self-BLEU':
        for index, row in df.iterrows():
            scores_list = eval(row[column_name])
            generated_text = row['generated']
            question_text = row['question']
            for score in scores_list:
                if score != 0: all_scores.append((score, generated_text, question_text))
        all_scores.sort(key=lambda x: x[0])
    else:
        for index, row in df.iterrows():
            scores_list = eval(row[column_name])
            generated_text = row['generated'].split('?, ')
            question_text = row['question']
            for score, generated in zip(scores_list, generated_text):
                if generated[-1] != '?': generated+='?'
                all_scores.append((score, generated, question_text))
        all_scores.sort(key=lambda x: x[0])

    return all_scores[:15]