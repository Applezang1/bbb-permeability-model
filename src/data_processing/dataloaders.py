from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd, torch
from transformers import PreTrainedTokenizerBase
from src.data_processing.dataset_fn import augment_data

        
def create_dataset(data: pd.DataFrame, 
                   val_split: float, 
                   random_seed: int): 
    '''
    Creates training and validation PyTorch datasets from the inputted Pandas DataFrame 

    Args: 
        data: The Pandas DataFrame specifically with a column of SMILES/SELFIES and 
        a column with BBB permeability label (name: labels)
        val_split: Percentage of data to be used for validating from the entire dataset
        random_seed: The random seed for the train_val split
        
    Returns: 
        Training and validation SMILES/SELFIES HuggingFace datasets
    ''' 

    # Load inputted Pandas DataFrame into a HuggingFace dataset
    dataset = Dataset.from_pandas(data) 

    # Split the dataset into training and validation datasets
    dataset = dataset.class_encode_column('labels')
    split_dataset = dataset.train_test_split(test_size=val_split,  
                                             seed=random_seed)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    return train_dataset, val_dataset


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



    






