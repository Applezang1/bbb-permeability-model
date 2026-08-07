from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd, torch
from transformers import PreTrainedTokenizerBase
from src.data_processing.dataset_fn import augment_data
        

def augment_dataset(train_dataset: Dataset,
                    num_augmentations: int, 
                    column_name: str, 
                    model_name: str):
    '''
    Augments the inputted training HuggingFace dataset 

    Args: 
        train_dataset: The training HuggingFace dataset to perform data augmentation on
        num_augmentations: The number of data augmentations to perform on the training
        dataset 
        column_name: The column name representing the chemicals in the dataset (SMILES/SELFIES)
        model_name: The name of the model that will be trained on this dataset
        
    Returns: 
        The augmented training HuggingFace dataset
    ''' 
    
    # Augment the inputted dataset 
    if num_augmentations != 0:
        train_dataset = augment_data(train_dataset, 
                                     num_augmentations, 
                                     column_name, 
                                     model_name)
        train_dataset = Dataset.from_pandas(train_dataset)   


    return train_dataset


def tokenize_dataset(dataset: Dataset,
                     tokenizer: PreTrainedTokenizerBase, 
                     column_name: str):
    '''
    Tokenizes the inputted HuggingFace dataset 

    Args: 
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer that will be used to tokenize the dataset 
        column_name: The column name representing the chemicals in the dataset (SMILES/SELFIES)
        
    Returns: 
        The tokenized PyTorch dataset
    ''' 

    # Tokenize the dataset
    def tokenization(batch): 
        '''Tokenizes inputs'''
        return tokenizer(batch[column_name], 
                         padding='max_length', 
                         truncation=True,
                         max_length=128,)
    
    dataset = dataset.map(tokenization, batched=True)

    # Remove unnecessary columns
    dataset = dataset.remove_columns(column_name)

    # Convert HuggingFace dataset to a PyTorch dataset
    dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    

    return dataset


def create_dataloader(dataset: torch.utils.data.Dataset,   
                      batch_size: int, 
                      shuffle: bool,
                      num_workers=0): 
    '''
    Converts a PyTorch dataset to a PyTorch dataloader
    
    Args: 
        dataset: PyTorch dataset to be converted
        batch_size: The size of each batch of data in the training and testing dataset 
        num_workers: Number of CPUs to dedicate to creating dataloaders
        shuffle: A boolean that dictates whether the dataloader is shuffled or not

    Returns: 
        A PyTorch dataloader    
    '''  

    # Create the training dataloader 
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True
    )


    return dataloader





    






