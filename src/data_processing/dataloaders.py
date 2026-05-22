from datasets import Dataset
from torch.utils.data import DataLoader


def create_smiles_dataset(data, test_validation_split, validation_split, tokenizer): 
    '''Creates training, testing, and validation PyTorch datasets from the SMILES Pandas DataFrame 

    Args: 
        tokenizer: Tokenizer to use for creating datasets 
        data: PyTorch DataFrame specifically with a column of SMILES (name: SMILES) and 
        a column with BBB permeability label (name: labels)
        test_validation_split: Percentage of data to be used for testing and validation 
        validation_split: Percentage of data to be used for validation from the testing and validation dataset
        
    Returns: 
        Training and testing SMILES PyTorch datasets
    ''' 

    # Load inputted Pandas DataFrame into a dataset
    dataset = Dataset.from_pandas(data) 
    
    # Tokenize the dataset
    def tokenization(dataset): 
        '''Tokenizes SMILES inputs'''
        return tokenizer(dataset['SMILES'], 
                         padding='max_length', 
                         truncation=True,
                         max_length=128,)
    
    dataset = dataset.map(tokenization, batched=True)
    dataset = dataset.remove_columns('SMILES')

    # Split the dataset into training and testing + validation datasets
    dataset = dataset.class_encode_column('labels')
    split_dataset = dataset.train_test_split(test_size=test_validation_split, 
                                             stratify_by_column='labels')
    train_dataset = split_dataset['train']

    # Split the dataset into a testing and validation dataset
    split_dataset = split_dataset['test'].train_test_split(test_size=validation_split, 
                                                   stratify_by_column='labels')
    test_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    # Convert HuggingFace dataset to a PyTorch dataset
    train_dataset.set_format(type='torch', 
                             columns=['input_ids', 'labels', 'attention_mask'])
    test_dataset.set_format(type='torch', 
                            columns=['input_ids', 'labels', 'attention_mask'])
    validation_dataset.set_format(type='torch', 
                            columns=['input_ids', 'labels', 'attention_mask'])
    

    return train_dataset, test_dataset, validation_dataset 


def create_selfies_dataset(data, test_validation_split, validation_split, tokenizer): 
    '''Creates training, testing, and validation PyTorch datasets from SELFIES Pandas DataFrame

    Args: 
        tokenizer: Tokenizer to use for creating datasets 
        data: PyTorch DataFrame specifically with a column of SELFIES (name: SELFIES) and 
        a column with BBB permeability label (name: labels)
        test_validation_split: Percentage of data to be used for testing and validation 
        validation_split: Percentage of data to be used for validation from the testing and validation dataset
        
    Returns: 
        Training and testing SELFIES PyTorch datasets
    ''' 

    # Load inputted Pandas DataFrame into a dataset
    dataset = Dataset.from_pandas(data) 
    
    # Tokenize the dataset
    def tokenization(dataset): 
        '''Tokenizes SELFIES inputs'''
        return tokenizer(dataset['SELFIES'], 
                         padding='max_length', 
                         truncation=True,
                         max_length=128,)
    
    dataset = dataset.map(tokenization, batched=True)
    dataset = dataset.remove_columns('SELFIES')

    # Split the dataset into training and testing + validation datasets
    dataset = dataset.class_encode_column('labels')
    split_dataset = dataset.train_test_split(test_size=test_validation_split, 
                                             stratify_by_column='labels')
    train_dataset = split_dataset['train']

    # Split the dataset into a testing and validation dataset
    split_dataset = split_dataset['test'].train_test_split(test_size=validation_split, 
                                                   stratify_by_column='labels')
    test_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    # Convert HuggingFace dataset to a PyTorch dataset
    train_dataset.set_format(type='torch', 
                             columns=['input_ids', 'labels', 'attention_mask'])
    test_dataset.set_format(type='torch', 
                            columns=['input_ids', 'labels', 'attention_mask'])
    validation_dataset.set_format(type='torch', 
                            columns=['input_ids', 'labels', 'attention_mask'])
    

    return train_dataset, test_dataset, validation_dataset


def create_dataloader(train_dataset, test_dataset, batch_size, num_workers=0): 
    '''Creates training and testing dataloaders
    
    Args: 
        train_dataset: Training dataset for the model
        test_dataset = Testing dataset for the model 
        batch_size = The size of each batch of data in the training and testing dataset 
        num_workers = Number of CPUs to dedicate to creating dataloaders

    Returns: 
        Training and testing PyTorch dataloaders    
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

    return train_dataloader, test_dataloader





    






