import torch, numpy as np, seaborn as sns, matplotlib.pyplot as plt, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


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
        The testing loss, mcc score, a confusion matrix of TP, FP, FN, TN, testing logits,
        testing prediction labels, and the testing labels
    '''

    # Put model in evaluation mode
    model.eval()

    # Initialize metric values
    test_loss = 0
    test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0 
    test_pred_label_list = []
    test_logits_list = []
    test_label_list = []

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

            # Store test logits and model prediction labels
            for logits in test_logit:
                test_logits_list.append(logits[1].detach().cpu().item())

            for pred_label in test_pred_label:
                test_pred_label_list.append(pred_label.detach().cpu().item())

            # Store test label in list
            for label in input['labels']:
                test_label_list.append(label.detach().cpu().item())

            for idx in range(len(test_pred_label)):
                if test_pred_label[idx] == 1 and input['labels'][idx] == 1: 
                    test_tp += 1 
                elif test_pred_label[idx] == 1 and input['labels'][idx] == 0: 
                    test_fp += 1 
                elif test_pred_label[idx] == 0 and input['labels'][idx] == 0: 
                    test_tn += 1 
                elif test_pred_label[idx] == 0 and input['labels'][idx] == 1: 
                    test_fn += 1

        # Calculate final validation loss and mcc score
        test_mcc = calculate_mcc(test_tp, test_tn, test_fp, test_fn) 
        test_loss = test_loss / len(test_dataloader) 
        test_confusion_matrix = [test_tp, test_tn, test_fp, test_fn]
        
    return test_loss, test_mcc, test_confusion_matrix, test_logits_list, test_pred_label_list, test_label_list


def raise_err():
    '''Raises a generic ValueError'''
    raise ValueError()


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


class EarlyStopping:
    """Early stops the training if validation MCC doesn't improve after a given patience."""
    def __init__(self, 
                 patience=5, 
                 delta=0, 
                 path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation MCC improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.counter = 0
        self.best_val_mcc = None
        self.early_stop = False
        self.val_mcc_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_mcc, model):
        if self.best_val_mcc is None:
            self.best_val_mcc = val_mcc
            self.save_checkpoint(val_mcc, model)
        elif val_mcc > self.best_val_mcc + self.delta:
            # Significant improvement detected
            self.best_val_mcc = val_mcc
            self.save_checkpoint(val_mcc, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_mcc, model):
        '''Saves model when validation MCC increases.'''
        torch.save(model.state_dict(), self.path)
        self.val_mcc_min = val_mcc


def minimize_with_rdkit(input_molfname,
                        sdf_out,
                        smi_col=None,
                        mol_name_col=None,
                        maxIters=400,
                        force_field="MMFF94s",
                        sep=r"\s+"):
    """
    Add hydrogen for 3D coordinates and minimize the geometry with RdKit.
    Function originates from B3clf
    """
    removed_molecule = 0
    removed_molecules_list = []
    # load molecules
    if input_molfname.lower().endswith(".smi") or input_molfname.lower().endswith(".csv"):
        df_mol = pd.read_csv(input_molfname, sep=sep, engine="python", header=None)
        if df_mol.shape[1] == 1:
            # Case for only SMILES column
            smile_list = df_mol.iloc[:, -1].to_list()
            mol_name_list = df_mol.iloc[:, -1].to_list()
        else:
            # Case for SMILES and MOL name columns
            if smi_col is None:
                smile_list = df_mol.iloc[:, 0].to_list()
            else:
                smile_list = df_mol[smi_col].to_list()

            if mol_name_col is None:
            # todo: use name if column name is valid
                mol_name_list = df_mol.iloc[:, -1].to_list()
            else:
                mol_name_list = df_mol[mol_name_col].to_list()

        mols = []
        for idx, smi in enumerate(smile_list):
            mol = Chem.MolFromSmiles(smi)
            # This will overwrite
            if mol is not None:
                mol.SetProp("_Name", mol_name_list[idx])
                mols.append(mol)

    writer = Chem.SDWriter(sdf_out)
    for idx, mol in enumerate(mols):
        new_mol = mol
        mol = Chem.AddHs(mol)
        if force_field == "MMFF94s":
            AllChem.EmbedMolecule(mol, randomSeed=999)
            # the following code will raise some errors
            try:
                mini_tag = AllChem.MMFFOptimizeMolecule(mol, force_field, maxIters=maxIters)
                # 0 optimize converged
                # -1 can not set up force field
                # 1 more iterations required
                if mini_tag == 0:
                     writer.write(mol)
                else:
                    if mini_tag == 1:
                        AllChem.MMFFOptimizeMolecule(mol, force_field, maxIters=maxIters * 2)
                    elif mini_tag == -1:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=400)
                    writer.write(mol)
                
            except (ValueError, RuntimeError) as e:
                print(f'Skipping {mol} due to Bad Conformer Id') 
                removed_molecule+=1
                removed_molecules_list.append(Chem.MolToSmiles(new_mol))
                continue

        elif force_field == "uff":
            # use uff force field if possible
            AllChem.EmbedMolecule(mol, randomSeed=999)
            # the following code will raise some errors
            try: 
                mini_tag = AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
                # 0 optimize converged
                # -1 can not set up force field
                # 1 more iterations required
                if mini_tag == 0:
                    writer.write(mol)
                else:
                    if mini_tag == 1:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters * 2)
                    elif mini_tag == -1:
                        AllChem.MMFFOptimizeMolecule(mol, "MMFF94s", maxIters=maxIters)
                    writer.write(mol)
            except (ValueError, RuntimeError) as e:
                print(f'Skipping {mol} due to Bad Conformer Id') 
                removed_molecule+=1
                removed_molecules_list.append(Chem.MolToSmiles(new_mol))
                continue

        else:
            raise NotImplementedError("This method is not implemented yet.")

    print(f'Number of compounds removed: {removed_molecule}')
    print(f'Removed molecules: {removed_molecules_list}')
    writer.close()


def count_classes(pred_dataframe: pd.DataFrame,
                  test_dataframe: list, 
                  pred_smi_column: str):
    '''
    Determines the SMILES that were filtered out of the testing dataframe for prediction and
    calculates the BBB+ and BBB- count of the filtered SMILES list
    
    Args: 
        pred_dataframe: The dataframe containing the curated SMILES and its respective label
        test_dataframe: The original testing dataframe containing SMILES (col_name: 'SMILES') and its respective label (col_name: 'labels')
        pred_smi_column: The name of the column that contains the SMILES in pred_dataframe
    
    Returns: 
        The BBB+ and BBB- count of the filtered SMILES list
    '''

    # Create a dataframe containing the filtered SMILES
    filtered_smiles_dataframe = test_dataframe[~test_dataframe['SMILES'].isin(pred_dataframe[pred_smi_column])].copy()

    # Initialize variables to store the BBB+ and BBB- count 
    positive_count, negative_count = 0, 0

    # Count the number of BBB+ and BBB- compounds: 
    for smiles in filtered_smiles_dataframe['SMILES']:
        if filtered_smiles_dataframe.loc[filtered_smiles_dataframe['SMILES'] == smiles, 'labels'].values == 1:
            positive_count += 1 
        elif filtered_smiles_dataframe.loc[filtered_smiles_dataframe['SMILES'] == smiles, 'labels'].values == 0: 
            negative_count += 1

    # Print the BBB+ and BBB- count
    print(f'{positive_count} BBB+ compounds were filtered from the testing dataset')
    print(f'{negative_count} BBB- compounds were filtered from the testing dataset')