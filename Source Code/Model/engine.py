import torch
from tqdm import tqdm

def train_step(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    '''
    Trains a PyTorch model for one epoch 

    Args: 
        model: The PyTorch model to train 
        train_dataloader: The PyTorch dataloader to train the model with 
        optimizer: The PyTorch optimizer used to minimize the loss function
        device: Device used for training

    Returns: 
        Training accuracy and training loss after one epoch
    
    '''
    # Put model in train mode
    model.train()
    train_loss = 0
    train_acc = 0 

    # Run training loop for each batch in the training dataloader
    for batch, input in enumerate(train_dataloader): 
        # Put data on target device
        input = {k: v.to(device) for k, v in input.items()} 

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

    # Calculate final training loss and accuracy
    train_acc = train_acc / len(train_dataloader)
    train_loss = train_loss / len(train_dataloader) 

    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              test_dataloader: torch.utils.data.DataLoader, 
              device: torch.device):
    '''
    Tests the model performance for one epoch

    Args: 
        model: The PyTorch model to train 
        test_dataloader: The Pytorch dataloader to test the model with
        device: Device used for training

    Returns: 
        The testing loss and accuracy for one epoch
    '''
    # Put the model in evaluation mode
    model.eval() 
    test_loss = 0 
    test_acc = 0

    # Run testing loop for each batch in the testing dataloader
    with torch.inference_mode():
        for batch, input in enumerate(test_dataloader): 
            # Put data on target device
            input = {k: v.to(device) for k, v in input.items()} 

            # Compute a forward pass 
            output = model(**input) 

            # Calculate the loss 
            loss = output.loss 
            test_loss += loss 

            # Calculate the accuracy 
            test_logits = output.logits 
            test_pred_label = test_logits.argmax(dim=1)
            test_acc += (test_pred_label == input['labels']).sum().item() / len(test_pred_label)
    
    # Calculate final testing loss and accuracy
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)

    return test_loss, test_acc 


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          device: torch.device, 
          num_epochs: int):
    
    '''
    Trains and tests the model performance for a given number of epochs 

    Args: 
        model: The PyTorch model to train 
        train_dataloader: The PyTorch dataloader to train the model with 
        test_dataloader: The Pytorch dataloader to test the model with
        optimizer: The PyTorch optimizer used to minimize the loss function
        device: Device used for training
        num_epochs: Number of epochs to train and test the model for

    Returns: 
        A 'results' dictionary with the train_loss, train_acc, test_loss, and test_acc over each epoch
    '''
    # Define the 'results' dictionary to store training and testing loss/accuracy
    results = {
    'train_loss': [], 
    'train_acc': [], 
    'test_loss': [],
    'test_acc': []
    }   

    for epoch in tqdm(range(num_epochs)):
        # Train step
        train_loss, train_acc = train_step(model, train_dataloader, optimizer, device)

        # Test step
        test_loss, test_acc = test_step(model, test_dataloader, device)

        # Print out what's happening
        print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.3f} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.3f}')

        # Add results to the 'results' dictionary
        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss.detach().cpu().numpy())
        results['test_acc'].append(test_acc) 

    return results

    




