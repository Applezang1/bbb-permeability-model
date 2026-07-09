import torch, yaml, argparse, optuna
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from src.data_processing.dataloaders import create_dataloader, augment_dataset, tokenize_dataset
from src.data_processing.dataset_fn import space_selfies_strings
from src.modeling.engine import train, train_step
from datasets import Dataset
from torchinfo import summary
from src.modeling.factory import create_model, dataset_loader
from src.utils import save_model, load_model, plot_confusion_matrix, test_on_testing_set
from src.modeling.tune import objective
from tqdm import tqdm


# Define arguments to run train.py
parser = argparse.ArgumentParser()
parser.add_argument('config_file', 
                    help='Type the .yaml path containing the model hyperparameters and information')
parser.add_argument('--save', 
                    action='store_true', 
                    help='Enable saving the model weights into saved_models')
parser.add_argument ('--validate', 
                     action='store_true', 
                     help='Enable the model to be tested on the final validation dataset using the MCC score')
parser.add_argument('--train', 
                    action='store_true', 
                    help='Enable model training and validation')
parser.add_argument('--full_train', 
                    action='store_true', 
                    help='Trains the model on 100 percent of the train_val DataFrame')
parser.add_argument('--tune', 
                    type=int, 
                    help='Specify the number of trials for hyperparameter tuning')
args = parser.parse_args()


# Initialize hyperparameters
with open(args.config_file, 'r') as hyp_file: 
    configs = yaml.safe_load(hyp_file)

TEST_SPLIT = configs['hyperparameters']['TEST_SPLIT']
BATCH_SIZE = configs['hyperparameters']['BATCH_SIZE']
NUM_WORKERS = configs['hyperparameters']['NUM_WORKERS']
NUM_EPOCHS = configs['hyperparameters']['NUM_EPOCHS']
LR = configs['hyperparameters']['LR']
NUM_AUGMENTATIONS = configs['hyperparameters']['NUM_AUGMENTATIONS']
WEIGHT_DECAY = configs['hyperparameters']['WEIGHT_DECAY']
NUM_SPLITS = configs['hyperparameters']['NUM_SPLITS']
BETA1 = configs['hyperparameters']['BETA1']
BETA2 = configs['hyperparameters']['BETA2']
CLASSIFIER_DROPOUT = configs['hyperparameters']['CLASSIFIER_DROPOUT']
column_name = dataset_loader(configs)
model_name = configs['model_information']['model_name']

# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Stratified K Fold Class
skf = StratifiedKFold(n_splits=NUM_SPLITS, 
                      shuffle=True, 
                      random_state=42)

# Instantiate model, optimizer, and tokenizer 
model, tokenizer = create_model(configs, 
                                CLASSIFIER_DROPOUT)
model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=LR, 
                              weight_decay=WEIGHT_DECAY, 
                              betas=(BETA1, BETA2))

# Initialize an empty dictionary to store model training results
final_results = {
    'train_mcc': [],
    'train_loss': [],
    'val_mcc': [],
    'val_loss': [],
}

### Load in train_val and testing datafranes ###
if column_name == 'SMILES':
    train_val_dataframe = pd.read_csv('data/smiles_train_val_dataset.csv')
    test_dataframe = pd.read_csv('data/smiles_test_dataset.csv')
elif column_name == 'SELFIES': 
    train_val_dataframe = pd.read_csv('data/selfies_train_val_dataset.csv')
    test_dataframe = pd.read_csv('data/selfies_test_dataset.csv')


### Create testing dataloader 
# Convert the testing Pandas DataFrames to HuggingFAce datasets
test_dataset = Dataset.from_pandas(test_dataframe)

# Implement required spacing to testing dataset's SELFIES strings if the model is SELFIES-TED
if model_name == 'SELFIES-TED':
    # Implement spacing to validation dataset
    test_dataset = test_dataset.to_pandas()
    test_dataset = space_selfies_strings(test_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

# Tokenize the testing dataset               
test_dataset = tokenize_dataset(test_dataset,
                                tokenizer, 
                                column_name)

# Create PyTorch testing dataloader
test_dataloader = create_dataloader(test_dataset, 
                                    BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)


### Optimize the hyperparameter based on the argument
if args.tune:
    # Maximize the val_mcc score during hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial,  
                                           configs, 
                                           train_val_dataframe,
                                           model_name,    
                                           device,
                                           column_name,
                                           NUM_AUGMENTATIONS,
                                           NUM_EPOCHS, 
                                           BATCH_SIZE, 
                                           NUM_WORKERS, 
                                           NUM_SPLITS), 
                                           n_trials=args.tune)
    best_lr = study.best_params['lr']
    best_weight_decay = study.best_params['weight_decay']
    best_beta1 = study.best_params['beta1']
    best_beta2 = study.best_params['beta2']
    best_classifier_dropout = study.best_params['classifier_dropout']

    # Print the best hyperparameter
    print(f'Best Learning rate: {best_lr}')
    print(f'Best Weight Decay: {best_weight_decay}')
    print(f'Best beta1 value: {best_beta1}')
    print(f'Best beta2 value: {best_beta2}')
    print(f'Best classifier dropout rate: {best_classifier_dropout}')

    # Plot visualizations
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig = optuna.visualization.plot_slice(study, params=['lr', 'weight_decay'])
    fig.show()


if args.train:
    ### Create training and validation dataloaders ###
    for fold, (train_index, test_index) in enumerate(skf.split(train_val_dataframe, train_val_dataframe['labels'])):

        # Reinstantate model optimizer, and tokenizer 
        model, tokenizer = create_model(configs, 
                                        CLASSIFIER_DROPOUT)
        model.to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), 
                                      lr=LR, 
                                      weight_decay=WEIGHT_DECAY, 
                                      betas=(BETA1, BETA2))
        
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

        # Augment the training dataset
        train_dataset = augment_dataset(train_dataset,  
                                        NUM_AUGMENTATIONS, 
                                        column_name, 
                                        model_name)

        # Tokenize the training and validation dataset               
        train_dataset = tokenize_dataset(train_dataset,
                                        tokenizer, 
                                        column_name)
        
        val_dataset = tokenize_dataset(val_dataset, 
                                       tokenizer, 
                                       column_name)

        # Create PyTorch training dataloader
        train_dataloader = create_dataloader(train_dataset, 
                                             BATCH_SIZE, 
                                             shuffle=True,
                                             num_workers=NUM_WORKERS)
        
        # Create PyTorch validation dataloader
        val_dataloader = create_dataloader(val_dataset, 
                                           BATCH_SIZE, 
                                           shuffle=False, 
                                           num_workers=NUM_WORKERS)


        ### Training ###
        # Train the model based on the argument
        results = train(model=model, 
                        fold=fold,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        optimizer=optimizer,
                        device=device,
                        num_epochs=NUM_EPOCHS)

        # Store train and validation metrics
        final_results['train_mcc'].append(results['train_mcc'])
        final_results['train_loss'].append(results['train_loss'])
        final_results['val_mcc'].append(results['val_mcc'])
        final_results['val_loss'].append(results['val_loss'])


if args.train:
    ### Plot Results ###
    num_epochs = np.arange(0, NUM_EPOCHS, 1)

    # Instantiate variables to store average metric values across folds 
    avg_train_mcc = []
    avg_train_loss = []
    avg_val_mcc = []
    avg_val_loss = []

    # Obtain the average metric values across folds
    train_mcc = np.array(final_results['train_mcc'])
    train_loss = np.array(final_results['train_loss'])
    val_mcc = np.array(final_results['val_mcc'])
    val_loss = np.array(final_results['val_loss'])

    for i in range(NUM_EPOCHS):
        avg_train_mcc.append(np.mean(train_mcc[:, i]))
        avg_train_loss.append(np.mean(train_loss[:, i]))
        avg_val_mcc.append(np.mean(val_mcc[:, i]))
        avg_val_loss.append(np.mean(val_loss[:, i]))

    # Plot the change in training and validation loss over epochs 
    fig = plt.figure(figsize=(8, 8))
    plt.plot(num_epochs, avg_train_loss, label='Train Loss')
    plt.plot(num_epochs, avg_val_loss, label='Val Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the change in training and validation mcc over epochs 
    fig = plt.figure(figsize=(8, 8))
    plt.plot(num_epochs, avg_train_mcc, label='Train MCC')
    plt.plot(num_epochs, avg_val_mcc, label='Val MCC')
    plt.ylabel('MCC')
    plt.xlabel('Number of Epochs')
    plt.legend()
    plt.savefig('mcc.png', dpi=300, bbox_inches='tight')
    plt.show()


# Train the model on 100% of the train_val dataset based on argument
if args.full_train:
    # Convert the train_val Pandas DataFrames to HuggingFace training dataset
    train_dataset = Dataset.from_pandas(train_val_dataframe)

    # Augment the training dataset
    train_dataset = augment_dataset(train_dataset,  
                                    NUM_AUGMENTATIONS, 
                                    column_name, 
                                    model_name)

    # Tokenize the training dataset               
    train_dataset = tokenize_dataset(train_dataset,
                                     tokenizer, 
                                     column_name)

    # Create PyTorch training dataloader
    train_dataloader = create_dataloader(train_dataset, 
                                         BATCH_SIZE, 
                                         shuffle=True,
                                         num_workers=NUM_WORKERS)
    
    ### Full Training Loop 
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_mcc = train_step(model, 
                                           train_dataloader, 
                                           optimizer, 
                                           device)
        tqdm.write(f' Epoch: {epoch} | Train Loss: {train_loss} | Train MCC: {train_mcc}')


# Save model weights into save_models based on argument
if args.save:
    model_name = configs['model_information']['model_name']
    save_model(model.state_dict(), model_name)


# Test the saved model on testing dataset based on argument
if args.validate: 
    model_name = configs['model_information']['model_name']
    saved_model = load_model(model, f'saved_models/{model_name}.pth')
    test_loss, test_mcc, test_confusion_matrix = test_on_testing_set(saved_model, device, test_dataloader)
    print(f'Testing Loss: {test_loss.item():.3f}')
    print(f'Testing MCC Score: {test_mcc:.3f}')
    print(f'Total Number of Data Points in Testing Dataset: {len(test_dataset)}')
    plot_confusion_matrix(test_confusion_matrix)


'''
# Create a summary of the model
summary(model=model, 
        input_size=(BATCH_SIZE, 512), 
        col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
        dtypes=[torch.long], 
        row_settings=['var_names'], 
        col_width=20)'''