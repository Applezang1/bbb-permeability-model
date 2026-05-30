import torch, yaml, argparse, optuna
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.data_processing.dataloaders import create_dataloader, create_dataset
from src.modeling.engine import train
from torchinfo import summary
from src.modeling.factory import create_model, dataset_loader
from src.utils import save_model, load_model, test_on_validation_set, plot_confusion_matrix 
from src.modeling.tune import objective


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
                    help='Enable model training')
parser.add_argument('--tune', 
                    type=int, 
                    help='Specify the number of trials for hyperparameter tuning')
args = parser.parse_args()


# Initialize hyperparameters
with open(args.config_file, 'r') as hyp_file: 
    configs = yaml.safe_load(hyp_file)

TEST_VALIDATION_SPLIT = configs['hyperparameters']['TEST_VALIDATION_SPLIT']
VALIDATION_SPLIT = configs['hyperparameters']['VALIDATION_SPLIT']
BATCH_SIZE = configs['hyperparameters']['BATCH_SIZE']
NUM_WORKERS = configs['hyperparameters']['NUM_WORKERS']
NUM_EPOCHS = configs['hyperparameters']['NUM_EPOCHS']
LR = configs['hyperparameters']['LR']


# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define model optimizer, and tokenizer 
model, tokenizer = create_model(configs)
model.to(device)

optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=LR)


### Curate BBB data ###
dataset = configs['dataset_information']['dataset']
BBB_data = pd.read_csv(dataset)


### Create training, testing, and validation datasets ###
column_name = dataset_loader(configs)
train_dataset, test_dataset, validation_dataset = create_dataset(BBB_data, 
                                                                 column_name,
                                                                 TEST_VALIDATION_SPLIT,
                                                                 VALIDATION_SPLIT, 
                                                                 tokenizer)


### Create training and testing dataloaders ###
train_dataloader, test_dataloader, validation_dataloader = create_dataloader(train_dataset, 
                                                                             test_dataset, 
                                                                             validation_dataset, 
                                                                             BATCH_SIZE, 
                                                                             NUM_WORKERS)


# Optimize the hyperparameter based on the argument
if args.tune:
    # Maximize the val_mcc score during hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, 
                                           model, 
                                           train_dataloader, 
                                           test_dataloader, 
                                           validation_dataloader, 
                                           optimizer, 
                                           device, 
                                           NUM_EPOCHS), 
                                        n_trials=args.tune)
    best_params = study.best_params

    # Print the best hyperparameter and plot visualizations
    print(f'Best Parameters: {study.best_params}')
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig = optuna.visualization.plot_slice(study, params=['lr'])
    fig.show()


### Training ###
# Train the model based on the argument
if args.train:
    results = train(model=model, 
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    device=device,
                    num_epochs=NUM_EPOCHS)


    ### Plot Results ###
    num_epochs = np.arange(0, len(results['train_loss']), 1)

    # Plot the change in training and testing loss over epochs 
    fig = plt.figure(figsize=(8, 8))
    plt.plot(num_epochs, results['train_loss'], label='Train Loss')
    plt.plot(num_epochs, results['test_loss'], label='Test Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.show()

    # Plot the change in training and testing mcc over epochs 
    fig = plt.figure(figsize=(8, 8))
    plt.plot(num_epochs, results['train_mcc'], label='Train MCC')
    plt.plot(num_epochs, results['test_mcc'], label='Test MCC')
    plt.ylabel('MCC')
    plt.xlabel('Number of Epochs')
    plt.legend()
    plt.show()


# Save model weights into save_models based on argument
if args.save:
    save_model(model.state_dict(), 'chemberta')


# Test the saved model on validation dataset based on argument
if args.validate: 
    saved_model = load_model(model, 'saved_models/chemberta.pth')
    val_loss, val_mcc, val_confusion_matrix = test_on_validation_set(saved_model, device, validation_dataloader)
    print(f'Validation Loss: {val_loss.item():.3f}')
    print(f'Validation MCC Score: {val_mcc:.3f}')
    print(f'Total Number of Data Points in Validation Dataset: {len(validation_dataset)}')
    plot_confusion_matrix(val_confusion_matrix)


'''
# Create a summary of the model
summary(model=model, 
        input_size=(BATCH_SIZE, 512), 
        col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
        dtypes=[torch.long], 
        row_settings=['var_names'], 
        col_width=20)'''
