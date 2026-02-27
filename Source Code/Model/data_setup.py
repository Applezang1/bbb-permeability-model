from datasets import Dataset
from torch.utils.data import random_split, DataLoader

def create_dataset(tokenizer, data, train_split, test_split): 
    '''Creates training and testing PyTorch datasets from Pandas DataFrame

    Args: 
        tokenizer: Tokenizer to use for creating datasets 
        data: PyTorch DataFrame specifically with a column of SMILES (name: SMILES) and 
        a column with BBB permeability label (name: BBclass)
        train_split: Percentage of data to be used for training 
        test_split: Percentage of data to be used for testing
        
    Returns: 
        Training and testing PyTorch datasets
    ''' 

    # Load inputted Pandas DataFrame into a dataset
    dataset = Dataset.from_pandas(data) 

    # Define a tokenization function to tokenize the SMILES string
    def tokenization(dataset): 
        return tokenizer(dataset['SMILES'], padding='max_length', truncation=True, max_length=200)
    
    # Tokenize the dataset
    dataset = dataset.map(tokenization, batched=True)
    dataset = dataset.remove_columns('SMILES')

    # Convert HuggingFace dataset to a PyTorch dataset
    dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])

    # Split the dataset into training and testing datasets
    train_dataset, test_dataset = random_split(dataset, [train_split, test_split])
    
    return train_dataset, test_dataset

def create_dataloader(train_dataset, test_dataset, batch_size, num_workers=0): 
    '''Creates training and testing dataloaders
    
    Args: 
        train_dataset: Training dataset for the model
        test_dataset = Training dataset for the model 
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
    num_workers=num_workers
    )

    # Create the testing dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers
    ) 

    return train_dataloader, test_dataloader





    






