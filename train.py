import torch, yaml, argparse
import numpy as np, pandas as pd
from src.data_processing.dataloaders import create_selfies_dataset, create_dataloader, create_smiles_dataset
from src.modeling.engine import train
from torchinfo import summary
import matplotlib.pyplot as plt
from src.modeling.factory import create_model, dataset_loader

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='Type the .yaml path containing the model hyperparameters and information')
args = parser.parse_args()

# Initialize hyperparameters
with open(args.config_file, 'r') as hyp_file: 
    hyperparameters = yaml.safe_load(hyp_file)

TEST_VALIDATION_SPLIT = hyperparameters['hyperparameters']['TEST_VALIDATION_SPLIT']
VALIDATION_SPLIT = hyperparameters['hyperparameters']['VALIDATION_SPLIT']
BATCH_SIZE = hyperparameters['hyperparameters']['BATCH_SIZE']
NUM_WORKERS = hyperparameters['hyperparameters']['NUM_WORKERS']
NUM_EPOCHS = hyperparameters['hyperparameters']['NUM_EPOCHS']
LR = hyperparameters['hyperparameters']['LR']

# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define model optimizer, and tokenizer 
with open(args.config_file, 'r') as model_file: 
    model_information = yaml.safe_load(model_file)

model, tokenizer = create_model(model_information)
model.to(device)

optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=LR)

### Curate BBB data ###
with open(args.config_file, 'r') as dataset_file: 
    dataset_information = yaml.safe_load(dataset_file)

dataset = dataset_information['dataset_information']['dataset']
BBB_data = pd.read_csv(dataset)

### Create training, testing, and validation datasets ###
create_dataset = dataset_loader(dataset_information)
train_dataset, test_dataset, validation_dataset = create_dataset(BBB_data, 
                                             TEST_VALIDATION_SPLIT,
                                             VALIDATION_SPLIT, 
                                             tokenizer)

### Create training and testing dataloaders ###
train_dataloader, test_dataloader = create_dataloader(train_dataset, 
                                                      test_dataset, 
                                                      BATCH_SIZE, 
                                                      NUM_WORKERS)


### Training ###
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


'''
# Create a summary of the model
summary(model=model, 
        input_size=(BATCH_SIZE, 512), 
        col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
        dtypes=[torch.long], 
        row_settings=['var_names'], 
        col_width=20)'''
