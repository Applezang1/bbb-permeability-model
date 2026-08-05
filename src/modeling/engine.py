import torch
from tqdm import tqdm
from src.utils import calculate_mcc


def train_step(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               device: torch.device, 
               lr_scheduler: torch.optim.lr_scheduler):
    '''
    Trains a PyTorch model for one epoch 

    Args: 
        model: The PyTorch model to train 
        train_dataloader: The PyTorch dataloader to train the model with 
        optimizer: The PyTorch optimizer used to minimize the loss function
        device: Device used for training

    Returns: 
        Training mcc score and training loss after one epoch
    
    '''
    # Put model in train mode
    model.train()
    train_loss = 0
    train_tp, train_tn, train_fp, train_fn = 0, 0, 0, 0 

    # Run training loop for each batch in the training dataloader
    for batch, input in enumerate(train_dataloader): 
        # Put data on target device
        input = {k: v.to(device) for k, v in input.items()} 

        # Compute a forward pass 
        output = model(**input) 

        # Calculate the loss 
        loss = output.loss 
        train_loss += loss

        # Calculate training mcc score 
        train_logit = output.logits 
        train_pred_label = train_logit.argmax(dim=1) 

        for idx in range(len(train_pred_label)):
            if train_pred_label[idx] == 1 and input['labels'][idx] == 1: 
                train_tp += 1 
            elif train_pred_label[idx] == 1 and input['labels'][idx] == 0: 
                train_fp += 1 
            elif train_pred_label[idx] == 0 and input['labels'][idx] == 0: 
                train_tn += 1 
            elif train_pred_label[idx] == 0 and input['labels'][idx] == 1: 
                train_fn += 1

        # Optimizer zero grad 
        optimizer.zero_grad()

        # Loss backwards 
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Scheduler step
        lr_scheduler.step()

    # Calculate final training loss and mcc score
    train_mcc = calculate_mcc(train_tp, train_tn, train_fp, train_fn)
    train_loss = train_loss / len(train_dataloader) 

    return train_loss, train_mcc


def val_step(model: torch.nn.Module, 
             val_dataloader: torch.utils.data.DataLoader, 
             device: torch.device):
    '''
    Validates the model performance for one epoch

    Args: 
        model: The PyTorch model to train 
        val_dataloader: The Pytorch dataloader to validate the model with
        device: Device used for training

    Returns: 
        The validation loss and mcc score for one epoch
    '''
    # Put the model in evaluation mode
    model.eval() 
    val_loss = 0 
    val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0

    # Run validation loop for each batch in the validation dataloader
    with torch.inference_mode():
        for batch, input in enumerate(val_dataloader): 
            # Put data on target device
            input = {k: v.to(device) for k, v in input.items()} 

            # Compute a forward pass 
            output = model(**input) 

            # Calculate the loss 
            loss = output.loss 
            val_loss += loss

            # Calculate the accuracy 
            val_logits = output.logits 
            val_pred_label = val_logits.argmax(dim=1)

            for idx in range(len(val_pred_label)):
                if val_pred_label[idx] == 1 and input['labels'][idx] == 1: 
                    val_tp += 1 
                elif val_pred_label[idx] == 1 and input['labels'][idx] == 0: 
                    val_fp += 1 
                elif val_pred_label[idx] == 0 and input['labels'][idx] == 0: 
                    val_tn += 1 
                elif val_pred_label[idx] == 0 and input['labels'][idx] == 1: 
                    val_fn += 1
        
    # Calculate final validation loss and accuracy
    val_loss = val_loss / len(val_dataloader)
    val_mcc = calculate_mcc(val_tp, val_tn, val_fp, val_fn)

    return val_loss, val_mcc 


def train(model: torch.nn.Module, 
          fold,
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          device: torch.device, 
          num_epochs: int):
    
    '''
    Trains and validates the model performance for a given number of epochs 

    Args: 
        model: The PyTorch model to train 
        fold: The current fold number of training
        train_dataloader: The PyTorch dataloader to train the model with 
        val_dataloader: The Pytorch dataloader to validate the model with
        optimizer: The PyTorch optimizer used to minimize the loss function
        device: Device used for training
        num_epochs: Number of epochs to train and validate the model for

    Returns: 
        A 'results' dictionary with the train_loss, train_acc, val_loss, and val_acc over each epoch
    '''
    # Define the 'results' dictionary to store training and validation loss/accuracy
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
        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_mcc'].append(train_mcc)
        results['val_loss'].append(val_loss.detach().cpu().numpy())
        results['val_mcc'].append(val_mcc) 

    return results

    




