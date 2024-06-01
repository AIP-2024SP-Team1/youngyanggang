import pandas as pd
from datasets import Dataset, DatasetDict

def preprocess_data(train_path, val_path, test1_path, test2_path, output_dir):
    # Load the Excel files
    train_data1 = pd.read_excel(train_path)
    train_data2 = pd.read_excel(train_path)
    val_data = pd.read_excel(val_path)
    test_data = pd.read_excel(test1_path)
    
    # Extract the necessary columns and rename them
    train_data1 = train_data1[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    train_data2 = train_data2[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    val_data = val_data[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    test_data = test_data[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    
    # Save the preprocessed data to CSV files
    train_data1.to_csv(f'{output_dir}/preprocessed_train1.csv', index=False)
    train_data2.to_csv(f'{output_dir}/preprocessed_train2.csv', index=False)
    val_data.to_csv(f'{output_dir}/preprocessed_val.csv', index=False)
    test_data.to_csv(f'{output_dir}/preprocessed_test.csv', index=False)
    
    # Convert to HuggingFace Datasets
    dataset_dict = DatasetDict({
        'train1': Dataset.from_pandas(train_data1),
        'train2': Dataset.from_pandas(train_data2),
        'validation': Dataset.from_pandas(val_data),
        'test': Dataset.from_pandas(test_data),
    })
    
    return dataset_dict

if __name__ == "__main__":
    train_path1 = '../../../data/ftqa_wh_2_train1.xlsx'
    train_path2 = '../../../data/ftqa_wh_2_train2.xlsx'
    val_path = '../../../data/ftqa_wh_2_val.xlsx'
    test_path = '../../../data/ftqa_wh_2_test.xlsx'
    output_dir = './'
    
    datasets = preprocess_data(train_path1, train_path2, val_path, test_path, output_dir)
    print("Preprocessing complete. Datasets:")
    print(datasets)
