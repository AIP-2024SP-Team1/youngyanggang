import pandas as pd
import json

file_path = './ftqa_wh_2_train1.xlsx'
df = pd.read_excel(file_path)

data_dict = {}
id=0

for _, row in df.iterrows():
    unique_key = (row['book_id'], row['section_id'])
    cor_section = row['cor_section']
    question = row['question']

    if unique_key in data_dict:
        data_dict[unique_key]['output'].append(question)
    else:
        data_dict[unique_key] = {
            "id": id,
            "instruction": "Generate appropriate multiple questions considering the context of the input. The interrogative word of the question should be one of the following seven: [what, when, where, which, who, why, how].",
            "input": cor_section,
            "output": [question],
            "book_name": row['book_name'],
            "section_id": row['section_id']
        }
        id+=1

json_data = [value for value in data_dict.values()]

for data in json_data:
    output = data['output']
    if type(output) == list:
        output = ", ".join(output)
    data['output'] = output

