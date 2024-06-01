import pandas as pd
from datasets import Dataset, DatasetDict

def preprocess_data(train_path, val_path, test1_path, test2_path, output_dir):
    # Load the Excel files
    train_data = pd.read_excel(train_path)
    val_data = pd.read_excel(val_path)
    test1_data = pd.read_excel(test1_path)
    test2_data = pd.read_excel(test2_path)
    
    # Extract the necessary columns and rename them
    train_data = train_data[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    val_data = val_data[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    test1_data = test1_data[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    test2_data = test2_data[['cor_section', 'question']].rename(columns={"cor_section": "context", "question": "question"})
    
    # Save the preprocessed data to CSV files
    train_data.to_csv(f'{output_dir}/preprocessed_train.csv', index=False)
    val_data.to_csv(f'{output_dir}/preprocessed_val.csv', index=False)
    test1_data.to_csv(f'{output_dir}/preprocessed_test1.csv', index=False)
    test2_data.to_csv(f'{output_dir}/preprocessed_test2.csv', index=False)
    
    # Convert to HuggingFace Datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_data),
        'validation': Dataset.from_pandas(val_data),
        'test1': Dataset.from_pandas(test1_data),
        'test2': Dataset.from_pandas(test2_data)
    })
    
    return dataset_dict

if __name__ == "__main__":
    train_path = '../../../data/ftqa_wh_2_train.xlsx'
    val_path = '../../../data/ftqa_wh_2_val.xlsx'
    test1_path = '../../../data/ftqa_wh_2_test1.xlsx'
    test2_path = '../../../data/ftqa_wh_2_test2.xlsx'
    output_dir = '../../../data'
    
    datasets = preprocess_data(train_path, val_path, test1_path, test2_path, output_dir)
    print("Preprocessing complete. Datasets:")
    print(datasets)
