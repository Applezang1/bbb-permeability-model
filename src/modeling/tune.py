import torch, pandas as pd, numpy as np
from tqdm import tqdm
from src.data_processing.dataloaders import create_dataloader, tokenize_dataset
from src.data_processing.dataset_fn import space_selfies_strings
from src.modeling.engine import train_step, val_step
from datasets import Dataset
from src.modeling.factory import create_model
from sklearn.model_selection import StratifiedKFold


def objective(trial, 
              config_file, 
              train_val_dataframe: pd.DataFrame, 
              model_name: str, 
              device: torch.device,
              column_name: str, 
              num_epochs: int, 
              batch_size: int, 
              num_workers: int,
              num_splits: int): 
    '''
    Define the hyperparameters that is being optimized (learning rate, weight decay, beta1, beta2, classifier dropout rate)
    as well as the model training and validation logic.

    Args: 
        trial: Optuna object that suggests hyperparameter values
        config_file: The configuration file containing the model information
        train_val_dataframe: The Pandas DataFrame to train and validate the model with 
        model_name: The name of the model whose hyperparameters will be optimized
        device: Device used for training and validation
        column_name: The column name representing the chemicals in the dataset (SMILES/SELFIES)
        num_epochs: The number of epochs to train and validate the model for
        batch_size: The size of each batch of data in the training and validation dataset 
        num_workers: Number of CPUs to dedicate to creating dataloaders
        num_splits: Number of times to split the train_val_dataloader


    Returns: 
        The final averaged val_mcc score after training on the proposed hyperparameter value with k-fold cross validation
    '''

    # Initialize Stratified K Fold Class
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Instantiate dictionary to store model results
    final_results = {
        'val_mcc': [],
    }

    # Define hyperparameter to be optimized
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.9, 0.95)
    beta2 = trial.suggest_float("beta2", 0.98, 0.9999, log=True)
    classifier_dropout = trial.suggest_float('classifier_dropout', 0.0, 0.5)


    ### Undergo K-Fold Cross Validation to analyze hyperparameter performance ###
    for fold, (train_index, test_index) in enumerate(skf.split(train_val_dataframe, train_val_dataframe['labels'])):

        # Reinstantate model, optimizer, and tokenizer 
        model, tokenizer = create_model(config_file, 
                                        classifier_dropout)
        model.to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), 
                                      lr=lr, 
                                      weight_decay=weight_decay, 
                                      betas=(beta1, beta2))
        
        # Split train_val dataframe into train and validation Pandas DataFrames
        train_dataset = train_val_dataframe.iloc[train_index, :]
        val_dataset = train_val_dataframe.iloc[test_index, :]

        # Convert the train and validation Pandas DataFrames to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)

        # Implement required spacing to validation dataset's SELFIES strings if the model is SELFIES-TED
        if model_name == 'SELFIES-TED':
            # Implement spacing to validation dataset
            val_dataset = val_dataset.to_pandas()
            val_dataset = space_selfies_strings(val_dataset)
            val_dataset = Dataset.from_pandas(val_dataset)

        # Implement required spacing to training dataset's SELFIES strings if the model is SELFIES-TED
        if model_name == 'SELFIES-TED':
            # Implement spacing to validation dataset
            train_dataset = train_dataset.to_pandas()
            train_dataset = space_selfies_strings(train_dataset)
            train_dataset = Dataset.from_pandas(train_dataset)
        
        # Tokenize the training and validation dataset               
        train_dataset = tokenize_dataset(train_dataset,
                                         tokenizer, 
                                         column_name)
        
        val_dataset = tokenize_dataset(val_dataset, 
                                       tokenizer, 
                                       column_name)

        # Create PyTorch training dataloader
        train_dataloader = create_dataloader(train_dataset, 
                                             batch_size,
                                             shuffle=True, 
                                             num_workers=num_workers)
        
        # Create PyTorch validation dataloader
        val_dataloader = create_dataloader(val_dataset, 
                                           batch_size, 
                                           shuffle=False,
                                           num_workers=num_workers)


        ### Training ###
        # Make a results dictionary to store final results
        results = {
            'train_loss': [], 
            'train_mcc': [], 
            'val_loss': [],
            'val_mcc': []
        }   

        for epoch in tqdm(range(num_epochs)):
            # Train step
            train_loss, train_mcc = train_step(model, train_dataloader, optimizer, device)

            # Validation step
            val_loss, val_mcc = val_step(model, val_dataloader, device)

            # Print out what's happening
            tqdm.write(f' Fold: {fold} | Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train MCC: {train_mcc:.3f} | Val Loss: {val_loss:.3f} | Val MCC: {val_mcc:.3f}')
            
            # Add results to the 'results' dictionary
            results['val_mcc'].append(val_mcc) 

        # Store validation metric
        final_results['val_mcc'].append(results['val_mcc'])

        trial.report(results['val_mcc'][-1], fold)
        if trial.should_prune():
            # Instantiate variables to store average metric value across folds 
            avg_val_mcc = []

            # Obtain the average metric values across folds
            val_mcc = np.array(final_results['val_mcc'])
                
            for i in range(num_epochs):
                avg_val_mcc.append(np.mean(val_mcc[:, i]))
                
            return avg_val_mcc[num_epochs-1]


    ### Report Final Val MCC Score ###
    # Instantiate variables to store average metric value across folds 
    avg_val_mcc = []

    # Obtain the average metric values across folds
    val_mcc = np.array(final_results['val_mcc'])
        
    for i in range(num_epochs):
        avg_val_mcc.append(np.mean(val_mcc[:, i]))
        
    return avg_val_mcc[num_epochs-1]