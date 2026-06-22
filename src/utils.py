import torch, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path


def save_model(model_parameters, 
               model_name: str):
    '''
    Saves model weights

    Args: 
        model_parameters: The model parameters needed to be saved   
        model_name: The name of the model for file naming during model saving    
    '''
    
    # Create model save path
    model_save_path = Path('saved_models')
    if model_save_path.is_dir(): 
        print(f'Path already exists')
    else: 
        print(f'Creating Path for Model Save')
        model_save_path.mkdir(parents=True, exist_ok=True)

    # Create a path under the model save path to store model parameters
    parameters_save_path = model_save_path / f'{model_name}.pth'

    # Save the model parameters
    torch.save(model_parameters, parameters_save_path) 


def load_model(model: torch.nn.Module, 
               weight_save_path: str):
    '''
    Loads saved model weights onto a PyTorch model

    Args: 
        model: The PyTorch model to load the saved weights onto 
        parameters_save_path: The path of the saved model weights

    Returns: 
        The loaded PyTorch model
    
    '''
    # Load the parameters in the inputted path onto a PyTorch model
    model.load_state_dict(torch.load(weight_save_path, weights_only=True))

    return model


def calculate_mcc(TP: int, 
                  TN: int, 
                  FP: int, 
                  FN: int): 
    '''
    Calculates the Matthews Correlation Coefficient 

    Args: 
        TN: true negative
        TP: true positive
        FP: false positive 
        FN: false negative 

    Returns: 
        The Matthews Correlation Coefficient Value
    '''

    mcc_numerator = (TP*TN - FP*FN)
    mcc_denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) 

    if TP+FP == 0 or TP+FN == 0 or TN+FP ==0 or TN+FN == 0: 
        mcc = 0
    else: 
        mcc = mcc_numerator/mcc_denominator

    return mcc


def test_on_testing_set(model: torch.nn.Module, 
                        device: torch.device, 
                        test_dataloader: torch.utils.data.DataLoader):
    '''
    Test model performance on the test dataloader

    Args: 
        model: The PyTorch model to test 
        testing_dataloader: The Pytorch dataloader to test the model with
        device: Device used for testing

    Returns: 
        The testing loss, mcc score, and a confusion matrix of TP, FP, FN, TN
    '''

    # Put model in evaluation mode
    model.eval()

    # Initialize metric values
    test_loss = 0
    test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0 

    # Run the testing loop for each batch in the testing dataloader
    with torch.inference_mode():
        for batch, input in enumerate(test_dataloader): 
            # Put data on target device
            input = {k: v.to(device) for k, v in input.items()} 

            # Compute a forward pass 
            output = model(**input) 

            # Calculate the testing loss 
            loss = output.loss 
            test_loss += loss

            # Calculate testing mcc score 
            test_logit = output.logits 
            test_pred_label = test_logit.argmax(dim=1) 

            for idx in range(len(test_pred_label)):
                if test_pred_label[idx] == 1 and input['labels'][idx] == 1: 
                    test_tp += 1 
                elif test_pred_label[idx] == 1 and input['labels'][idx] == 0: 
                    test_fp += 1 
                elif test_pred_label[idx] == 0 and input['labels'][idx] == 0: 
                    test_tn += 1 
                elif test_pred_label[idx] == 0 and input['labels'][idx] == 1: 
                    test_fn += 1

        # Calculate final validattion loss and mcc score
        test_mcc = calculate_mcc(test_tp, test_tn, test_fp, test_fn) 
        test_loss = test_loss / len(test_dataloader) 
        test_confusion_matrix = [test_tp, test_tn, test_fp, test_fn]
        
    return test_loss, test_mcc, test_confusion_matrix


def plot_confusion_matrix(confusion_matrix_values: list[int]):
    '''
    Plots a 2x2 Confusion Matrix of TN, TP, FP, FN. 
        TN: true negative
        TP: true positive
        FP: false positive 
        FN: false negative 

    Args: 
        confusion_matrix_values: A list of confusion matrix values in the 
        following order: TP, TN, FP, FN

    Returns: 
        The plotted 2x2 Confusion Matrix
    '''
    
    # Unpack TP, TN, FP, FN from the confusion matrix list
    tp, tn, fp, fn = confusion_matrix_values

    # Plot the confusion matrix
    sns.heatmap([[tp, fp], 
                 [fn, tn]], 
                annot=[[tp, fp], 
                       [fn, tn]], 
                xticklabels=['Positive (1)', 'Negative (0)'],
                yticklabels=['Positive (1)', 'Negative (0)'], 
                fmt=".0f", 
                cmap='Blues', 
                linewidths=1.5
                )
    plt.ylabel('Predicted Values')
    plt.xlabel('Actual Values')
    plt.title('Testing Confusion Matrix')
    plt.show()


