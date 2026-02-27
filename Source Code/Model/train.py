import torch
from dataset import BBB_data
from tqdm import tqdm
from data_setup import create_dataset, create_dataloader
from torchinfo import summary
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification


# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define ChemBERTa-2 and its tokenizer
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-77M-MTR"
                                                         ,num_labels=2
                                                         ,use_safetensors=True).to(device)

# Create training and testing datasets
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2
train_dataset, test_dataset = create_dataset(tokenizer, BBB_data, TRAIN_SPLIT, TEST_SPLIT)

# Create training and testing dataloaders 
BATCH_SIZE = 32 
NUM_WORKERS = 0

train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset 
                                                      ,BATCH_SIZE, NUM_WORKERS)

# Print information about the shapes of each element in the training dataloader
batch = next(iter(train_dataloader))
print(f"Input IDs shape in training dataloader: {batch['input_ids'][0].shape}")
print(f"Attention mask shape in training dataloader: {batch['attention_mask'][0].shape}")
print(f"Label shape in training dataloader: {batch['labels'][0].shape}") 

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# Define training step
epochs = 2

results = {
    'train_loss': [], 
    'train_acc': [], 
    'test_loss': [],
    'test_acc': []
}

for epoch in tqdm(range(epochs)):
    ### Training 
    model.train()
    train_loss = 0
    train_acc = 0 

    for batch, input in enumerate(train_dataloader): 
        # Compute a forward pass 
        output = model(**input) 

        # Calculate the loss 
        loss = output.loss 
        train_loss += loss

        # Calculate training accuracy 
        train_logit = output.logits 
        train_pred_label = train_logit.argmax(dim=1) 
        train_acc += (train_pred_label == input['labels']).sum().item() / len(train_pred_label)

        # Optimizer zero grad 
        optimizer.zero_grad()

        # Loss backwards 
        loss.backward()

        # Optimizer step
        optimizer.step()

    train_acc = train_acc / len(train_dataloader)
    train_loss = train_loss / len(train_dataloader) 

    ### Testing 
    model.eval() 
    test_loss = 0 
    test_acc = 0
    with torch.inference_mode():
        for batch, input in enumerate(test_dataloader): 
            # Compute a forward pass 
            output = model(**input) 

            # Calculate the loss 
            loss = output.loss 
            test_loss += loss 

            # Calculate the accuracy 
            test_logits = output.logits 
            test_pred_label = test_logits.argmax(dim=1)
            test_acc += (test_pred_label == input['labels']).sum().item() / len(test_pred_label)
    
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)

    # Add values to dictionary 
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

    # Print out what's happening 
    print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.3f} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.3f}')



print(results)
# Create a summary of the model
'''summary(model=model, 
        input_size=(BATCH_SIZE, 512), 
        col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
        dtypes=[torch.long], 
        row_settings=['var_names'], 
        col_width=20)'''