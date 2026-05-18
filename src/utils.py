import torch, numpy as np
from pathlib import Path

def save_model(model_parameters, model_name):
    '''
    Saves model weights

    Args: 
        model_parameters: The model parameters needed to be saved   
        model_name: The name of the model for file naming during model saving    
    '''
    
    # Create model save path
    model_save_path = Path('model_save')
    if model_save_path.is_dir(): 
        print(f'Path already exists')
    else: 
        print(f'Creating Path for Model Save')
        model_save_path.mkdir(parents=True, exist_ok=True)

    # Create a path under the model save path to store model parameters
    parameters_save_path = model_save_path / f'{model_name}.pth'

    # Save the model parameters
    torch.save(model_parameters, parameters_save_path) 


def load_model(model, weight_save_path):
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


def calculate_mcc(TP, TN, FP, FN): 
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


