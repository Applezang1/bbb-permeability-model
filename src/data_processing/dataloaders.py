from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd, torch
from transformers import PreTrainedTokenizerBase
from src.data_processing.dataset_fn import augment_dataset


def create_dataset(data: pd.DataFrame, 
                   column_name: str,
                   test_validation_split: float, 
                   validation_split: float, 
                   tokenizer: PreTrainedTokenizerBase, 
                   num_augmentations: int): 
    '''
    Creates training, testing, and validation PyTorch datasets from the inputted Pandas DataFrame 

    Args: 
        tokenizer: Tokenizer to use for creating datasets 
        column_name: The column name of the inputted PyTorch DataFrame for molecules (SMILES/SELFIES)
        data: Pandas DataFrame specifically with a column of SMILES/SELFIES and 
        a column with BBB permeability label (name: labels)
        test_validation_split: Percentage of data to be used for testing and validation 
        validation_split: Percentage of data to be used for validation from the testing and validation dataset
        num_augmentations: The number of times to perform data augmentation on the training dataset
        
    Returns: 
        Training, testing, and validation SMILES/SELFIES PyTorch datasets
    ''' 

    # Load inputted Pandas DataFrame into a dataset
    dataset = Dataset.from_pandas(data) 

    # Split the dataset into training and testing + validation datasets
    dataset = dataset.class_encode_column('labels')
    split_dataset = dataset.train_test_split(test_size=test_validation_split, 
                                             stratify_by_column='labels')
    train_dataset = split_dataset['train']

    # Augment the training dataset 
    train_dataset = augment_dataset(train_dataset, 
                                    num_augmentations, 
                                    column_name)
    train_dataset = Dataset.from_pandas(train_dataset)

    # Split the testing + validation dataset into a testing and validation dataset
    split_dataset = split_dataset['test'].train_test_split(test_size=validation_split, 
                                                           stratify_by_column='labels')
    test_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    # Tokenize the dataset
    def tokenization(batch): 
        '''Tokenizes inputs'''
        return tokenizer(batch[column_name], 
                         padding='max_length', 
                         truncation=True,
                         max_length=128,)
    
    train_dataset = train_dataset.map(tokenization, batched=True)
    test_dataset = test_dataset.map(tokenization, batched=True)
    validation_dataset = validation_dataset.map(tokenization, batched=True)

    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(column_name)
    test_dataset = test_dataset.remove_columns(column_name)
    validation_dataset = validation_dataset.remove_columns(column_name)

    # Convert HuggingFace dataset to a PyTorch dataset
    train_dataset.set_format(type='torch', 
                             columns=['input_ids', 'labels', 'attention_mask'])
    test_dataset.set_format(type='torch', 
                            columns=['input_ids', 'labels', 'attention_mask'])
    validation_dataset.set_format(type='torch', 
                            columns=['input_ids', 'labels', 'attention_mask'])
    

    return train_dataset, test_dataset, validation_dataset 


def create_dataloader(train_dataset: torch.utils.data.Dataset, 
                      test_dataset: torch.utils.data.Dataset, 
                      validation_dataset: torch.utils.data.Dataset, 
                      batch_size: int, 
                      num_workers=0): 
    '''Creates training, testing, and validation PyTorch dataloaders
    
    Args: 
        train_dataset: Training dataset for the model
        test_dataset: Testing dataset for the model 
        validation_dataset: Validation dataset for the model
        batch_size: The size of each batch of data in the training and testing dataset 
        num_workers: Number of CPUs to dedicate to creating dataloaders

    Returns: 
        Training, testing, and validation PyTorch dataloaders    
    '''  

    # Create the training dataloader 
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=False
    )

    # Create the testing dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=False
    ) 

    # Create the validation dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=False
    ) 

    return train_dataloader, test_dataloader, validation_dataloader





    






