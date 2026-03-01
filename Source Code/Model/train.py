import torch
import numpy as np
from data_setup import create_dataset, create_dataloader
from dataset import curate_bbb_data
from engine import train
from torchinfo import summary
import matplotlib.pyplot as plt
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification

# Initialize hyperparameters
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2
BATCH_SIZE = 32 
NUM_WORKERS = 0
NUM_EPOCHS = 50

# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define model and optimizer 
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-77M-MTR"
                                                         ,num_labels=2
                                                         ,use_safetensors=True).to(device)
optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=0.01)

### Curate BBB data ###
BBB_data = curate_bbb_data()

### Create training and testing datasets ###
train_dataset, test_dataset = create_dataset(BBB_data, 
                                             TRAIN_SPLIT, 
                                             TEST_SPLIT)

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

# Plot the change in training and testing accuracy over epochs 
fig = plt.figure(figsize=(8, 8))
plt.plot(num_epochs, results['train_acc'], label='Train Accuracy')
plt.plot(num_epochs, results['test_acc'], label='Test Accuracy')
plt.ylabel('Accuracy')
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
