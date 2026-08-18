import torch, yaml, argparse, optuna, statistics, sys, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from src.data_processing.dataloaders import create_dataloader, tokenize_dataset, create_dataset
from src.data_processing.dataset_fn import space_selfies_strings
from src.modeling.engine import train_step, val_step
from datasets import Dataset
from torchinfo import summary
from src.modeling.factory import create_model, dataset_loader
from src.utils import save_model, load_model, plot_confusion_matrix, test_on_testing_set, EarlyStopping
from src.modeling.tune import objective
from tqdm import tqdm
from transformers import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    average_precision_score
)


# Define arguments to run train.py
parser = argparse.ArgumentParser()
parser.add_argument('config_file', 
                    help='Type the .yaml path containing the model hyperparameters and information')
parser.add_argument('--save', 
                    action='store_true', 
                    help='Enable saving the model weights into saved_models')
parser.add_argument ('--test', 
                     action='store_true', 
                     help='Enable the model to be tested on the testing dataset')
parser.add_argument('--train', 
                    action='store_true', 
                    help='Enable model training and validation')
parser.add_argument('--tune', 
                    type=int, 
                    help='Specify the number of trials for hyperparameter tuning')
parser.add_argument('--external_test', 
                    action='store_true', 
                    help='Enable the model to be tested on the external testing dataset')
parser.add_argument('--validate', 
                    action='store_true', 
                    help='Enable the model to be tested on the validation dataset')
parser.add_argument('--k_fold', 
                    action='store_true', 
                    help='Enable model k-fold cross validation')
parser.add_argument('--k_fold_test', 
                    action='store_true', 
                    help='Test the k-fold cross validated models on the testing and external testing dataset')
args = parser.parse_args()


# Initialize hyperparameters
with open(args.config_file, 'r') as hyp_file: 
    configs = yaml.safe_load(hyp_file)

PATIENCE = configs['hyperparameters']['PATIENCE']
VAL_SPLIT = configs['hyperparameters']['VAL_SPLIT']
BATCH_SIZE = configs['hyperparameters']['BATCH_SIZE']
NUM_WORKERS = configs['hyperparameters']['NUM_WORKERS']
NUM_EPOCHS = configs['hyperparameters']['NUM_EPOCHS']
LR = configs['hyperparameters']['LR']
WEIGHT_DECAY = configs['hyperparameters']['WEIGHT_DECAY']
NUM_SPLITS = configs['hyperparameters']['NUM_SPLITS']
BETA1 = configs['hyperparameters']['BETA1']
BETA2 = configs['hyperparameters']['BETA2']
CLASSIFIER_DROPOUT = configs['hyperparameters']['CLASSIFIER_DROPOUT']
NUM_SPLITS = configs['hyperparameters']['NUM_SPLITS']
column_name = dataset_loader(configs)
model_name = configs['model_information']['model_name']

# Set verbosity to hide info, warnings, and progress bar
logging.set_verbosity_error()
logging.disable_progress_bar()

# Suppress standard library cleanup warnings during shutdown
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=ResourceWarning)

# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate model, optimizer, and tokenizer 
model, tokenizer = create_model(configs, 
                                CLASSIFIER_DROPOUT)
model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=LR, 
                              weight_decay=WEIGHT_DECAY, 
                              betas=(BETA1, BETA2))

### Load in train_val, testing, and external testing datafranes ###
if column_name == 'SMILES':
    train_val_dataframe = pd.read_csv('data/processed/smiles_train_dataframe.csv')
    test_dataframe = pd.read_csv('data/processed/smiles_test_dataframe.csv')
    external_test_dataframe = pd.read_csv('data/processed/smiles_external_test_dataframe.csv')

elif column_name == 'SELFIES': 
    train_val_dataframe = pd.read_csv('data/processed/selfies_train_dataframe.csv')
    test_dataframe = pd.read_csv('data/processed/selfies_test_dataframe.csv')
    external_test_dataframe = pd.read_csv('data/processed/selfies_external_test_dataframe.csv')


### Optimize the hyperparameter based on the argument
if args.tune:
    # Maximize the val_mcc score during hyperparameter optimization
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.15)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: objective(trial,  
                                           configs, 
                                           train_val_dataframe,
                                           model_name,    
                                           device,
                                           column_name,
                                           NUM_EPOCHS, 
                                           BATCH_SIZE, 
                                           NUM_WORKERS, 
                                           NUM_SPLITS), 
                                           n_trials=args.tune)
    best_lr = study.best_params['lr']
    best_weight_decay = study.best_params['weight_decay']
    best_beta1 = study.best_params['beta1']
    best_beta2 = study.best_params['beta2']
    best_classifier_dropout = study.best_params['classifier_dropout']

    # Print the best hyperparameter
    print(f'Best Learning rate: {best_lr}')
    print(f'Best Weight Decay: {best_weight_decay}')
    print(f'Best beta1 value: {best_beta1}')
    print(f'Best beta2 value: {best_beta2}')
    print(f'Best classifier dropout rate: {best_classifier_dropout}')

    # Plot visualizations
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig = optuna.visualization.plot_slice(study, 
                                          params=['lr', 'weight_decay', 'beta1', 'beta2', 'classifier_dropout'])
    fig.show()


# Implement K-Fold cross validation depending on argument
if args.k_fold: 
    # Define Stratified K-Fold Cross Validation Object
    skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)

    # Implement K-Fold cross validation
    for fold, (train_index, test_index) in enumerate(skf.split(train_val_dataframe, train_val_dataframe['labels'])):
        # Define early stopping object
        early_stopping = EarlyStopping(patience=PATIENCE,
                                       path=f'saved_models/{model_name}-{fold}.pth')
    
        # Reinstantate model, optimizer, and tokenizer 
        model, tokenizer = create_model(configs, 
                                        CLASSIFIER_DROPOUT)
        model.to(device)
    
        optimizer = torch.optim.AdamW(params=model.parameters(), 
                                      lr=LR, 
                                      weight_decay=WEIGHT_DECAY, 
                                      betas=(BETA1, BETA2))
            
        # Split train_val dataframe into train and validation Pandas DataFrames
        train_dataset = train_val_dataframe.iloc[train_index, :]
        val_dataset = train_val_dataframe.iloc[test_index, :]
    
        # Convert the train and validation Pandas DataFrames to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)
    
        # Implement required spacing to validation dataset's SELFIES strings if the model is SELFIES-TED
        if model_name == 'SELFIES-TED':
            # Implement spacing to validation dataset
            val_dataset = val_dataset.to_pandas()
            val_dataset = space_selfies_strings(val_dataset)
            val_dataset = Dataset.from_pandas(val_dataset)
    
        # Implement required spacing to training dataset's SELFIES strings if the model is SELFIES-TED
        if model_name == 'SELFIES-TED':
            # Implement spacing to validation dataset
            train_dataset = train_dataset.to_pandas()
            train_dataset = space_selfies_strings(train_dataset)
            train_dataset = Dataset.from_pandas(train_dataset)
            
        # Tokenize the training and validation dataset               
        train_dataset = tokenize_dataset(train_dataset,
                                         tokenizer, 
                                         column_name)
            
        val_dataset = tokenize_dataset(val_dataset, 
                                       tokenizer, 
                                       column_name)
    
        # Create PyTorch training dataloader
        train_dataloader = create_dataloader(train_dataset, 
                                             BATCH_SIZE,
                                             shuffle=True, 
                                             num_workers=NUM_WORKERS)
            
        # Create PyTorch validation dataloader
        val_dataloader = create_dataloader(val_dataset, 
                                           BATCH_SIZE, 
                                           shuffle=False,
                                           num_workers=NUM_WORKERS)

        # Define decay learning rate scheduler 
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                            start_factor=1, 
                                                            end_factor=0.01,
                                                            total_iters=25*len(train_dataloader))
        
        # Define warmup learning rate scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                             start_factor=0.01, 
                                                             end_factor=1,
                                                             total_iters=10*len(train_dataloader))
        
        # Concatenate warmup and decay learning rate schedulers 
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, 
                                                             schedulers=[warmup_scheduler, decay_scheduler], 
                                                             milestones=[10*len(train_dataloader)])
    
    
        ### Training ###
        for epoch in tqdm(range(NUM_EPOCHS)):
            # Train step
            train_loss, train_mcc = train_step(model, train_dataloader, optimizer, device, lr_scheduler)
    
            # Validation step
            val_loss, val_mcc = val_step(model, val_dataloader, device)
    
            # Print out what's happening
            tqdm.write(f' Fold: {fold} | Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train MCC: {train_mcc:.3f} | Val Loss: {val_loss:.3f} | Val MCC: {val_mcc:.3f}')
                
            early_stopping(val_mcc, model)
            if early_stopping.early_stop:
                print(" Early stopping triggered")
                break


# Test the K-Fold cross validated models on the TITAN-BBB and DrugBank datasets based on the argument
if args.k_fold_test:

    # Initialize model metric lists for validation results
    val_mcc_list = []
    val_accuracy_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_score_list = []
    val_auc_roc_list = []
    val_specificity_list = []
    val_auprc_list = []

    # Initialize model metric lists for TITAN-BBB results
    test_mcc_list = []
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_score_list = []
    test_auc_roc_list = []
    test_specificity_list = []
    test_auprc_list = []

    # Initialize model metric lists for DrugBank results
    ext_test_mcc_list = []
    ext_test_accuracy_list = []
    ext_test_precision_list = []
    ext_test_recall_list = []
    ext_test_f1_score_list = []
    ext_test_auc_roc_list = []
    ext_test_specificity_list = []
    ext_test_auprc_list = []
    ext_test_tp_list = []
    ext_test_tn_list = []
    ext_test_fp_list = []
    ext_test_fn_list = []

    ### Create Testing Dataloader ###
    # Convert test DataFrame into a HuggingFace Dataset
    test_dataset = Dataset.from_pandas(test_dataframe)
            
    # Implement required spacing to testing dataset's SELFIES strings if the model is SELFIES-TED
    if model_name == 'SELFIES-TED':
        # Implement spacing to testing dataset
        test_dataset = test_dataset.to_pandas()
        test_dataset = space_selfies_strings(test_dataset)
        test_dataset = Dataset.from_pandas(test_dataset)
            
    # Tokenize the testing dataset               
    test_dataset = tokenize_dataset(test_dataset,
                                    tokenizer, 
                                    column_name)
            
    # Create PyTorch testing dataloader
    test_dataloader = create_dataloader(test_dataset, 
                                        BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

    ### Create External Testing Dataloader ###
    # Convert the external testing Pandas DataFrames to HuggingFace datasets
    external_test_dataset = Dataset.from_pandas(external_test_dataframe)
    
    # Implement required spacing to the external testing dataset's SELFIES strings if the model is SELFIES-TED
    if model_name == 'SELFIES-TED':
        # Implement spacing to external testing dataset
        external_test_dataset = external_test_dataset.to_pandas()
        external_test_dataset = space_selfies_strings(external_test_dataset)
        external_test_dataset = Dataset.from_pandas(external_test_dataset)
    
    # Tokenize the external testing dataset               
    external_test_dataset = tokenize_dataset(external_test_dataset,
                                             tokenizer, 
                                             column_name)
    
    # Create PyTorch external testing dataloader
    external_test_dataloader = create_dataloader(external_test_dataset, 
                                                 BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=NUM_WORKERS)
    
    ### Test the model on its respective validation testing, and external testing dataloader ###
    # Define Stratified K-Fold Cross Validation Object
    skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)
    
    # Test models on their respective validation splits
    for fold, (train_index, test_index) in enumerate(skf.split(train_val_dataframe, train_val_dataframe['labels'])):
        # Initialize model
        saved_model = load_model(model, f'saved_models/{model_name}-{fold}.pth')

        # Split train_val dataframe into train and validation Pandas DataFrames
        train_dataset = train_val_dataframe.iloc[train_index, :]
        val_dataset = train_val_dataframe.iloc[test_index, :]
            
        # Convert the validation Pandas DataFrames to HuggingFace datasets
        val_dataset = Dataset.from_pandas(val_dataset)
            
        # Implement required spacing to validation dataset's SELFIES strings if the model is SELFIES-TED
        if model_name == 'SELFIES-TED':
            # Implement spacing to validation dataset
            val_dataset = val_dataset.to_pandas()
            val_dataset = space_selfies_strings(val_dataset)
            val_dataset = Dataset.from_pandas(val_dataset)
        
        # Tokenize the validation dataset
        val_dataset = tokenize_dataset(val_dataset, 
                                        tokenizer, 
                                        column_name)
                    
        # Create PyTorch validation dataloader
        val_dataloader = create_dataloader(val_dataset, 
                                        BATCH_SIZE, 
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

        # Run each model through its respective validation dataset
        val_loss, val_mcc, val_confusion_matrix, val_logits, val_pred_labels, val_labels = test_on_testing_set(saved_model, 
                                                                                                               device, 
                                                                                                               val_dataloader)
        
        # Run each model through the TITAN-BBB dataset
        test_loss, test_mcc, test_confusion_matrix, test_logits, test_pred_labels, test_labels = test_on_testing_set(saved_model, 
                                                                                                                     device, 
                                                                                                                     test_dataloader)
    
        # Run each model through the DrugBank dataset
        ext_test_loss, ext_test_mcc, ext_test_confusion_matrix, ext_test_logits, ext_test_pred_labels, ext_test_labels = test_on_testing_set(saved_model, 
                                                                                                                                             device, 
                                                                                                                                             external_test_dataloader)
    
        # Calculate and store model metrics for the validation dataset
        val_mcc_list.append(val_mcc)
        val_accuracy_list.append(accuracy_score(val_labels, val_pred_labels))
        val_precision_list.append(precision_score(val_labels, val_pred_labels))
        val_recall_list.append(recall_score(val_labels, val_pred_labels))
        val_f1_score_list.append(f1_score(val_labels, val_pred_labels))
        val_auc_roc_list.append(roc_auc_score(val_labels, val_logits))
        val_specificity_list.append(recall_score(val_labels, val_pred_labels, pos_label=0))
        val_auprc_list.append(average_precision_score(val_labels, val_pred_labels))

        # Calculate and store model metrics for the TITAN-BBB dataset
        test_mcc_list.append(test_mcc)
        test_accuracy_list.append(accuracy_score(test_labels, test_pred_labels))
        test_precision_list.append(precision_score(test_labels, test_pred_labels))
        test_recall_list.append(recall_score(test_labels, test_pred_labels))
        test_f1_score_list.append(f1_score(test_labels, test_pred_labels))
        test_auc_roc_list.append(roc_auc_score(test_labels, test_logits))
        test_specificity_list.append(recall_score(test_labels, test_pred_labels, pos_label=0))
        test_auprc_list.append(average_precision_score(test_labels, test_pred_labels))
    
        # Calculate and store model metrics for the DrugBank dataset
        ext_test_mcc_list.append(ext_test_mcc)
        ext_test_accuracy_list.append(accuracy_score(ext_test_labels, ext_test_pred_labels))
        ext_test_precision_list.append(precision_score(ext_test_labels, ext_test_pred_labels))
        ext_test_recall_list.append(recall_score(ext_test_labels, ext_test_pred_labels))
        ext_test_f1_score_list.append(f1_score(ext_test_labels, ext_test_pred_labels))
        ext_test_auc_roc_list.append(roc_auc_score(ext_test_labels, ext_test_logits))
        ext_test_specificity_list.append(recall_score(ext_test_labels, ext_test_pred_labels, pos_label=0))
        ext_test_auprc_list.append(average_precision_score(ext_test_labels, ext_test_pred_labels))
        ext_test_tp_list.append(ext_test_confusion_matrix[0])
        ext_test_tn_list.append(ext_test_confusion_matrix[1])
        ext_test_fp_list.append(ext_test_confusion_matrix[2])
        ext_test_fn_list.append(ext_test_confusion_matrix[3])
    
    # Calculate the mean of the model metrics across the 20 models for the validation dataset
    print(f'Validation MCC: {statistics.mean(val_mcc_list):.3f} +/- {statistics.pstdev(val_mcc_list):.3f}')
    print(f'Validation Accuracy: {statistics.mean(val_accuracy_list):.3f} +/- {statistics.pstdev(val_accuracy_list):.3f}')
    print(f'Validation Precision: {statistics.mean(val_precision_list):.3f} +/- {statistics.pstdev(val_precision_list):.3f}')
    print(f'Validation Recall: {statistics.mean(val_recall_list):.3f} +/- {statistics.pstdev(val_recall_list):.3f}')
    print(f'Validation F1 Score: {statistics.mean(val_f1_score_list):.3f} +/- {statistics.pstdev(val_f1_score_list):.3f}')
    print(f'Validation AUC-ROC: {statistics.mean(val_auc_roc_list):.3f} +/- {statistics.pstdev(val_auc_roc_list):.3f}')
    print(f'Validation Specificity: {statistics.mean(val_specificity_list):.3f} +/- {statistics.pstdev(val_specificity_list):.3f}')
    print(f'Validation AUPRC: {statistics.mean(val_auprc_list):.3f} +/- {statistics.pstdev(val_auprc_list):.3f}')
    print('')

    # Calculate the mean of the model metrics across the 20 models for TITAN-BBB
    print(f'Testing MCC: {statistics.mean(test_mcc_list):.3f} +/- {statistics.pstdev(test_mcc_list):.3f}')
    print(f'Testing Accuracy: {statistics.mean(test_accuracy_list):.3f} +/- {statistics.pstdev(test_accuracy_list):.3f}')
    print(f'Testing Precision: {statistics.mean(test_precision_list):.3f} +/- {statistics.pstdev(test_precision_list):.3f}')
    print(f'Testing Recall: {statistics.mean(test_recall_list):.3f} +/- {statistics.pstdev(test_recall_list):.3f}')
    print(f'Testing F1 Score: {statistics.mean(test_f1_score_list):.3f} +/- {statistics.pstdev(test_f1_score_list):.3f}')
    print(f'Testing AUC-ROC: {statistics.mean(test_auc_roc_list):.3f} +/- {statistics.pstdev(test_auc_roc_list):.3f}')
    print(f'Testing Specificity: {statistics.mean(test_specificity_list):.3f} +/- {statistics.pstdev(test_specificity_list):.3f}')
    print(f'Testing AUPRC: {statistics.mean(test_auprc_list):.3f} +/- {statistics.pstdev(test_auprc_list):.3f}')
    print('')
    
    # Calculate the mean of the model metrics across the 20 models for DrugBank
    print(f'External Testing MCC: {statistics.mean(ext_test_mcc_list):.3f} +/- {statistics.pstdev(ext_test_mcc_list):.3f}')
    print(f'External Testing Accuracy: {statistics.mean(ext_test_accuracy_list):.3f} +/- {statistics.pstdev(ext_test_accuracy_list):.3f}')
    print(f'External Testing Precision: {statistics.mean(ext_test_precision_list):.3f} +/- {statistics.pstdev(ext_test_precision_list):.3f}')
    print(f'External Testing Recall: {statistics.mean(ext_test_recall_list):.3f} +/- {statistics.pstdev(ext_test_recall_list):.3f}')
    print(f'External Testing F1 Score: {statistics.mean(ext_test_f1_score_list):.3f} +/- {statistics.pstdev(ext_test_f1_score_list):.3f}')
    print(f'External Testing AUC-ROC: {statistics.mean(ext_test_auc_roc_list):.3f} +/- {statistics.pstdev(ext_test_auc_roc_list):.3f}')
    print(f'External Testing Specificity: {statistics.mean(ext_test_specificity_list):.3f} +/- {statistics.pstdev(ext_test_specificity_list):.3f}')
    print(f'External Testing AUPRC: {statistics.mean(ext_test_auprc_list):.3f} +/- {statistics.pstdev(ext_test_auprc_list):.3f}')
    print(f'External Testing TP: {statistics.mean(ext_test_tp_list):.3f} +/- {statistics.pstdev(ext_test_tp_list):.3f}')
    print(f'External Testing TN: {statistics.mean(ext_test_tn_list):.3f} +/- {statistics.pstdev(ext_test_tn_list):.3f}')
    print(f'External Testing FP: {statistics.mean(ext_test_fp_list):.3f} +/- {statistics.pstdev(ext_test_fp_list):.3f}')
    print(f'External Testing FN: {statistics.mean(ext_test_fn_list):.3f} +/- {statistics.pstdev(ext_test_fn_list):.3f}')
    plot_confusion_matrix([statistics.mean(ext_test_tp_list), statistics.mean(ext_test_tn_list), statistics.mean(ext_test_fp_list), statistics.mean(ext_test_fn_list)])
                    

### Train the model on the train_val dataset based on argument
if args.train:
    # Define dictionary to store results 
    results = {
        'train_loss': [], 
        'train_mcc': [],
        'val_loss': [], 
        'val_mcc': []
    }

    # Define early stopping object
    early_stopping = EarlyStopping(patience=PATIENCE,
                                   path=f'saved_models/{model_name}-Ver2.pth')

    # Split the train_val Pandas DataFrame into training and validation datasets
    train_dataset, val_dataset = create_dataset(train_val_dataframe, 
                                                VAL_SPLIT,
                                                random_seed=42)

    # Implement required spacing to validation dataset's SELFIES strings if the model is SELFIES-TED
    if model_name == 'SELFIES-TED':
        # Implement spacing to validation dataset
        val_dataset = val_dataset.to_pandas()
        val_dataset = space_selfies_strings(val_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)

    # Implement required spacing to training dataset's SELFIES strings if the model is SELFIES-TED
    if model_name == 'SELFIES-TED':
        # Implement spacing to validation dataset
        train_dataset = train_dataset.to_pandas()
        train_dataset = space_selfies_strings(train_dataset)
        train_dataset = Dataset.from_pandas(train_dataset)

    # Tokenize the training dataset               
    train_dataset = tokenize_dataset(train_dataset,
                                     tokenizer, 
                                     column_name)
    
    # Tokenize the validation dataset 
    val_dataset = tokenize_dataset(val_dataset, 
                                   tokenizer, 
                                   column_name)


    # Create PyTorch training dataloader
    train_dataloader = create_dataloader(train_dataset, 
                                         BATCH_SIZE, 
                                         shuffle=True,
                                         num_workers=NUM_WORKERS)
    
    # Create PyTorch validation dataloader 
    val_dataloader = create_dataloader(val_dataset,
                                       BATCH_SIZE,
                                       shuffle=False, 
                                       num_workers=NUM_WORKERS)

    # Define decay learning rate scheduler 
    decay_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                        start_factor=1, 
                                                        end_factor=0.01,
                                                        total_iters=25*len(train_dataloader))

    # Define warmup learning rate scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                         start_factor=0.01, 
                                                         end_factor=1,
                                                         total_iters=10*len(train_dataloader))

    # Concatenate warmup and decay learning rate schedulers 
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, 
                                                         schedulers=[warmup_scheduler, decay_scheduler], 
                                                         milestones=[10*len(train_dataloader)])

    
    ### Full Training Loop 
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_mcc = train_step(model, 
                                           train_dataloader, 
                                           optimizer, 
                                           device, 
                                           lr_scheduler)
        val_loss, val_mcc = val_step(model, 
                                     val_dataloader, 
                                     device)
        tqdm.write(f' Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train MCC: {train_mcc:.3f} | Val Loss: {val_loss:.3f} | Val MCC: {val_mcc:.3f}')

        early_stopping(val_mcc, model)
        if early_stopping.early_stop:
            print(" Early stopping triggered")
            break

        # Append results 
        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_mcc'].append(train_mcc)
        results['val_loss'].append(val_loss.detach().cpu().numpy())
        results['val_mcc'].append(val_mcc) 

    ### Plot Results ###
    num_epochs = np.arange(1, epoch+1, 1)

    # Plot the change in training and validation loss over epochs 
    fig = plt.figure(figsize=(8, 8))
    plt.plot(num_epochs, results['train_loss'], label='Train Loss')
    plt.plot(num_epochs, results['val_loss'], label='Val Loss')
    plt.axvline(x=epoch-PATIENCE+1, color='red', linestyle='--', label='Early Stopping Checkpoint')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the change in training and validation mcc over epochs 
    fig = plt.figure(figsize=(8, 8))
    plt.plot(num_epochs, results['train_mcc'], label='Train MCC')
    plt.plot(num_epochs, results['val_mcc'], label='Val MCC')
    plt.axvline(x=epoch-PATIENCE+1, color='red', linestyle='--', label='Early Stopping Checkpoint')
    plt.ylabel('MCC')
    plt.xlabel('Number of Epochs')
    plt.legend()
    plt.savefig('mcc.png', dpi=300, bbox_inches='tight')
    plt.show()


# Save model weights into save_models based on argument
if args.save:
    model_name = configs['model_information']['model_name']
    save_model(model.state_dict(), model_name)


# Test the saved model on the validation dataset based on argument
if args.validate:
    # Load model information
    model_name = configs['model_information']['model_name']
    saved_model = load_model(model, f'saved_models/{model_name}-Ver2.pth')
    
    # Split the train_val Pandas DataFrame into training and validation datasets
    train_dataset, val_dataset = create_dataset(train_val_dataframe, 
                                                VAL_SPLIT,
                                                random_seed=42)

    # Implement required spacing to validation dataset's SELFIES strings if the model is SELFIES-TED
    if model_name == 'SELFIES-TED':
        # Implement spacing to validation dataset
        val_dataset = val_dataset.to_pandas()
        val_dataset = space_selfies_strings(val_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)

    # Tokenize the validation dataset 
    val_dataset = tokenize_dataset(val_dataset, 
                                    tokenizer, 
                                    column_name)
            
    # Create PyTorch validation dataloader 
    val_dataloader = create_dataloader(val_dataset,
                                        BATCH_SIZE,
                                        shuffle=False, 
                                        num_workers=NUM_WORKERS)

    # Compute and print model results on validation dataset
    val_loss, val_mcc, val_confusion_matrix, val_logits, val_pred_labels, val_labels = test_on_testing_set(saved_model, device, val_dataloader)
    print(f'Validation Loss: {val_loss.item():.3f}')
    print(f'Validation MCC Score: {val_mcc:.3f}')
    print(f'Accuracy: {accuracy_score(val_labels, val_pred_labels):.3f}')
    print(f'Precision: {precision_score(val_labels, val_pred_labels):.3f}')
    print(f'Recall: {recall_score(val_labels, val_pred_labels):.3f}')
    print(f'F1 Score: {f1_score(val_labels, val_pred_labels):.3f}')
    print(f'AUC-ROC: {roc_auc_score(val_labels, val_logits):.3f}')
    print(f'Specificity: {recall_score(val_labels, val_pred_labels, pos_label=0):.3f}')
    print(f'AUPRC: {average_precision_score(val_labels, val_pred_labels):.3f}')
    print(f'Total Number of Data Points in Testing Dataset: {len(val_dataset)}')
    plot_confusion_matrix(val_confusion_matrix)
    

# Test the saved model on testing dataset based on argument
if args.test: 
    # Load model information
    model_name = configs['model_information']['model_name']
    saved_model = load_model(model, f'saved_models/{model_name}-Ver2.pth')

    # Initialize empty lists to store model metrics
    accuracy_list = []
    mcc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_roc_list = []
    specificity_list = []
    auprc_list = []

    # Test model by subsampling 50% of the testing dataset 50 times
    for seed in range(50):
        # Convert the testing Pandas DataFrames to HuggingFace datasets
        test_data = test_dataframe.sample(frac=0.5, replace=False, random_state=seed)
        test_dataset = Dataset.from_pandas(test_data)

        # Implement required spacing to testing dataset's SELFIES strings if the model is SELFIES-TED
        if model_name == 'SELFIES-TED':
            # Implement spacing to testing dataset
            test_dataset = test_dataset.to_pandas()
            test_dataset = space_selfies_strings(test_dataset)
            test_dataset = Dataset.from_pandas(test_dataset)

        # Tokenize the testing dataset               
        test_dataset = tokenize_dataset(test_dataset,
                                        tokenizer, 
                                        column_name)

        # Create PyTorch testing dataloader
        test_dataloader = create_dataloader(test_dataset, 
                                            BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS)

        # Compute the model's results on the testing dataset split and store in a list
        test_loss, test_mcc, test_confusion_matrix, test_logits, test_pred_labels, test_labels = test_on_testing_set(saved_model, device, test_dataloader)
        accuracy_list.append(accuracy_score(test_labels, test_pred_labels))
        precision_list.append(precision_score(test_labels, test_pred_labels))
        mcc_list.append(test_mcc)
        recall_list.append(recall_score(test_labels, test_pred_labels))
        f1_list.append(f1_score(test_labels, test_pred_labels))
        auc_roc_list.append(roc_auc_score(test_labels, test_logits))
        specificity_list.append(recall_score(test_labels, test_pred_labels, pos_label=0))
        auprc_list.append(average_precision_score(test_labels, test_pred_labels))

    metrics_list = [
        accuracy_list, 
        precision_list, 
        mcc_list, 
        recall_list, 
        f1_list, 
        auc_roc_list, 
        specificity_list,
        auprc_list
    ]

    metrics_name = [
        'Accuracy', 
        'Precision', 
        'MCC', 
        'Recall', 
        'F1 Score', 
        'AUC-ROC', 
        'Specificity',
        'AUPRC'
    ]

    # Calculate final testing results
    for idx in range(len(metrics_list)):
        # Calculate mean
        mean_val = statistics.mean(metrics_list[idx])

        # Calculate Sample Variance (divides by n - 1)
        standard_deviation = statistics.pstdev(metrics_list[idx])

        print(f'Testing {metrics_name[idx]}: {mean_val:.3f} +/- {standard_deviation:.3f}')


# Test the saved model on the external testing dataset based on argument
if args.external_test:
    # Convert the external testing Pandas DataFrames to HuggingFace datasets
    external_test_dataset = Dataset.from_pandas(external_test_dataframe)

    # Implement required spacing to the external testing dataset's SELFIES strings if the model is SELFIES-TED
    if model_name == 'SELFIES-TED':
        # Implement spacing to external testing dataset
        external_test_dataset = external_test_dataset.to_pandas()
        external_test_dataset = space_selfies_strings(external_test_dataset)
        external_test_dataset = Dataset.from_pandas(external_test_dataset)

    # Tokenize the external testing dataset               
    external_test_dataset = tokenize_dataset(external_test_dataset,
                                            tokenizer, 
                                            column_name)

    # Create PyTorch external testing dataloader
    external_test_dataloader = create_dataloader(external_test_dataset, 
                                                BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=NUM_WORKERS)

    # Print external testing dataset results
    model_name = configs['model_information']['model_name']
    saved_model = load_model(model, f'saved_models/{model_name}-Ver2.pth')
    ext_test_loss, ext_test_mcc, ext_test_confusion_matrix, ext_test_logits, ext_test_pred_labels, ext_test_labels = test_on_testing_set(saved_model, device, external_test_dataloader)
    print(f'External Testing Loss: {ext_test_loss.item():.3f}')
    print(f'External Testing MCC Score: {ext_test_mcc:.3f}')
    print(f'Accuracy: {accuracy_score(ext_test_labels, ext_test_pred_labels):.3f}')
    print(f'Precision: {precision_score(ext_test_labels, ext_test_pred_labels):.3f}')
    print(f'Recall: {recall_score(ext_test_labels, ext_test_pred_labels):.3f}')
    print(f'F1 Score: {f1_score(ext_test_labels, ext_test_pred_labels):.3f}')
    print(f'AUC-ROC: {roc_auc_score(ext_test_labels, ext_test_logits):.3f}')
    print(f'Specificity: {recall_score(ext_test_labels, ext_test_pred_labels, pos_label=0):.3f}')
    print(f'AUPRC: {average_precision_score(test_labels, test_pred_labels):.3f}')
    print(f'Total Number of Data Points in Testing Dataset: {len(external_test_dataset)}')
    plot_confusion_matrix(ext_test_confusion_matrix)


'''
# Create a summary of the model
summary(model=model, 
        input_size=(BATCH_SIZE, 512), 
        col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
        dtypes=[torch.long], 
        row_settings=['var_names'], 
        col_width=20)'''