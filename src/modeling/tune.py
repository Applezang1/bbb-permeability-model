import optuna, torch
from src.utils import test_on_validation_set
from src.modeling.factory import create_model


def objective(trial, 
              config_file, 
              train_dataloader: torch.utils.data.DataLoader, 
              test_dataloader: torch.utils.data.DataLoader, 
              validation_dataloader: torch.utils.data.DataLoader, 
              device: torch.device, 
              num_epochs: int): 
    '''
    Define the hyperparameter that is being optimized (learning rate) as well as the model 
    training, testing, and validation logic.

    Args: 
        trial: Optuna object that suggests hyperparameter values
        config_file: The configuration file containing the model information
        train_dataloader: The PyTorch dataloader to train the model with 
        test_dataloader: The PyTorch dataloader to test the model with 
        validation_dataloader: The PyTorch dataloader to validate the model with 
        device: Device used for training and testing
        num_epochs: The number of epochs to train and test the model for

    Returns: 
        The final val_mcc score after training and testing on the proposed hyperparameter value
    '''

    # Instantiate model
    model, tokenizer = create_model(config_file)
    model.to(device)

    # Define hyperparameter to be optimized
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=lr)

    # Hyperparameter Tuning
    for epoch in range(num_epochs):
        ### Training ###
        # Put model in train mode
        model.train()
        train_loss = 0

        # Run training loop for each batch in the training dataloader
        for batch, input in enumerate(train_dataloader): 
            # Put data on target device
            input = {k: v.to(device) for k, v in input.items()} 

            # Compute a forward pass 
            output = model(**input) 

            # Calculate the loss 
            loss = output.loss 
            train_loss += loss

            # Optimizer zero grad 
            optimizer.zero_grad()

            # Loss backwards 
            loss.backward()

            # Optimizer step
            optimizer.step()

        # Calculate final training loss
        train_loss = train_loss / len(train_dataloader) 

        ### Testing ###
        model.eval() 
        test_loss = 0 

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
            
        # Calculate final testing loss and accuracy
        test_loss = test_loss / len(test_dataloader)

        val_loss, val_mcc, val_confusion_matrix = test_on_validation_set(model, device, validation_dataloader)
        trial.report(val_mcc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    final_val_loss, final_val_mcc, final_val_confusion_matrix = test_on_validation_set(model, device, validation_dataloader)


    return final_val_mcc
