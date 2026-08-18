import pandas as pd, argparse
from rdkit import Chem
from rdkit.Chem import PandasTools
from src.utils import plot_confusion_matrix, raise_err
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix, 
    average_precision_score
)


# Define arguments to run train.py
parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_bbb', 
                    action='store_true', 
                    help='Obtain DrugBank testing results for EnsembleBBB')
parser.add_argument('--deepred_bbb', 
                    action='store_true', 
                    help='Obtain DrugBank testing results for DeePred-BBB')
parser.add_argument('--b3clf', 
                    action='store_true', 
                    help='Obtain DrugBank testing results for B3clf')
parser.add_argument('--deep_b3', 
                    action='store_true', 
                    help='Obtain DrugBank testing results for Deep-B3')
args = parser.parse_args()


# Define DrugBank testing dataframe 
external_test_dataframe = pd.read_csv('data/processed/smiles_external_test_dataframe.csv')

# Obtain EnsembleBBB DrugBank testing results
ensemble_bbb_results = pd.read_csv('data/benchmark/output/ensemblebbb_prediction_results_drugbank.csv')

# Obtain DeePred-BBB DrugBank testing results
deepred_bbb_results = pd.read_csv('data/benchmark/output/DeePred-BBB_predictions_drugbank.csv')

# Obtain B3clf DrugBank testing results
b3clf_results = pd.read_excel('data/benchmark/output/b3clf_predictions_drugbank.xlsx')

# Obtain Deep-B3 DrugBank testing results 
deep_b3_results = pd.read_csv('data/benchmark/output/deep-b3_predictions_drugbank.csv')


### Calculate external testing results for DeePred-BBB based on the argument ###
if args.deepred_bbb:
    # Obtain DeePred-BBB training dataset 
    deepred_bbb_train_dataset = pd.read_excel('data/benchmark/train/deepred-bbb_train_dataset.XLSX')
    
    # Filter DeePred-BBB testing results to exclude training data
    overlapping_smiles_count = len(deepred_bbb_results[deepred_bbb_results['Name'].isin(deepred_bbb_train_dataset['Compounds'])])
    deepred_bbb_results = deepred_bbb_results[~deepred_bbb_results['Name'].isin(deepred_bbb_train_dataset['Compounds'])].copy()
    
    # Remove compounds that were unable to be predicted
    unpredictable_compounds = deepred_bbb_results.loc[deepred_bbb_results['Predicted_class'] == -2147483648]
    
    deepred_bbb_results = deepred_bbb_results[~deepred_bbb_results['Name'].isin(unpredictable_compounds['Name'].values)].copy()
    
    # Merge DeePred-BBB and DrugBank testing dataframe
    deepred_bbb_merged_dataframe = pd.merge(
        left=external_test_dataframe, 
        right=deepred_bbb_results, 
        how='right', 
        left_on='SMILES', 
        right_on='Name'
    )
    
    # Obtain prediction and true labels
    deepred_predicted_labels = deepred_bbb_merged_dataframe['Predicted_class'].values
    test_labels = deepred_bbb_merged_dataframe['labels'].values

    # Calculate TP, TN, FN, FP
    tn, fp, fn, tp = confusion_matrix(test_labels, deepred_predicted_labels).ravel()

    # Print DeePred-BBB testing metrics
    print('DeePred-BBB Metrics:')
    print(f'{overlapping_smiles_count} compounds from the DrugBank Database were removed due to overlap with training dataset')
    print(f'{len(unpredictable_compounds)} compounds from the DrugBank Database were unable to be predicted by the model')
    print(f'{len(external_test_dataframe['SMILES'])-len(deepred_bbb_results)} compounds were filtered from the DrugBank database in total')
    print('')
    print(f'MCC Score: {matthews_corrcoef(test_labels, deepred_predicted_labels):.3f}')
    print(f'Accuracy: {accuracy_score(test_labels, deepred_predicted_labels):.3f}')
    print(f'Precision: {precision_score(test_labels, deepred_predicted_labels):.3f}')
    print(f'Recall: {recall_score(test_labels, deepred_predicted_labels):.3f}')
    print(f'F1 Score: {f1_score(test_labels, deepred_predicted_labels):.3f}')
    print(f'Specificity: {recall_score(test_labels, deepred_predicted_labels, pos_label=0):.3f}')
    print(f'AUPRC: {average_precision_score(test_labels, deepred_predicted_labels):.3f}')
    print('')

    # Plot DeePred-BBB confusion matrix
    plot_confusion_matrix([tp, tn, fp, fn])


### Calcuate external testing results for EnsembleBBB ###
if args.ensemble_bbb:
    # Obtain EnsembleBBB training dataset 
    ensemble_bbb_train_dataset = pd.read_csv('data/benchmark/train/ensemble_bbb_train_dataset.tsv', 
                                             usecols=['SMILES', 'BBB+/BBB-'],
                                             sep='\t')

    # Convert smiles to canonical form
    invalid_smiles_list = []
    for idx, smiles in enumerate(ensemble_bbb_train_dataset['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            ensemble_bbb_train_dataset.at[idx, 'SMILES'] = canonical_smiles
        else:
            invalid_smiles_list.append(smiles)

    # Remove invalid SMILES from training dataset
    invalid_smiles_count = len(invalid_smiles_list)
    ensemble_bbb_train_dataset = ensemble_bbb_train_dataset[~ensemble_bbb_train_dataset['SMILES'].isin(invalid_smiles_list)].copy()

    # Remove redundant compounds in EnsembleBBB training dataset
    duplicate_dataframe = ensemble_bbb_train_dataset['SMILES'].value_counts()
    duplicate_dataframe = duplicate_dataframe[duplicate_dataframe>1]
    duplicate_SMILES_list = list(duplicate_dataframe.index)
    duplicate_smiles_count = len(duplicate_SMILES_list)
    ensemble_bbb_train_dataset = ensemble_bbb_train_dataset[~ensemble_bbb_train_dataset['SMILES'].isin(duplicate_SMILES_list)].copy()

    # Filter DeePred-BBB testing results to exclude training data
    overlapping_smiles_count = len(ensemble_bbb_results[ensemble_bbb_results['Smiles'].isin(ensemble_bbb_train_dataset['SMILES'])])
    ensemble_bbb_results = ensemble_bbb_results[~ensemble_bbb_results['Smiles'].isin(ensemble_bbb_train_dataset['SMILES'])].copy()

    # Merge EnsembleBBB and DrugBank testing dataframes 
    ensemble_bbb_merged_dataframe = pd.merge(
        left=ensemble_bbb_results, 
        right=external_test_dataframe, 
        left_on='Smiles', 
        right_on='SMILES', 
        how='left'
    )
    
    # Obtain prediction and true labels
    ensemble_predicted_results = ensemble_bbb_merged_dataframe['BBB+/BBB-']
    ensemble_predicted_results = [1 if ensemble_predicted_results[idx] == 'BBB+' else (0 if ensemble_predicted_results[idx] == 'BBB-' else raise_err()) for idx in range(len(ensemble_predicted_results))]
    test_labels = ensemble_bbb_merged_dataframe['labels'].values

    # Calculate TP, TN, FN, FP
    tn, fp, fn, tp = confusion_matrix(test_labels, ensemble_predicted_results).ravel()

    # Print EnsembleBBB testing metrics
    print('EnsembleBBB Metrics:')
    print('')
    print(f'{invalid_smiles_count} compounds were removed from the EnsembleBBB training dataset due to invalid structures/charges')
    print(f'{duplicate_smiles_count} compounds were removed from the EnsembleBBB training dataset due to redundancy')
    print(f'{overlapping_smiles_count} compounds were removed from the DrugBank Database due to overlaps with the training dataset')
    print(f'{len(external_test_dataframe['labels'])-len(ensemble_predicted_results)} compounds were filtered from the DrugBank Database in total')
    print('')
    print(f'MCC Score: {matthews_corrcoef(test_labels, ensemble_predicted_results):.3f}')
    print(f'Accuracy: {accuracy_score(test_labels, ensemble_predicted_results):.3f}')
    print(f'Precision: {precision_score(test_labels, ensemble_predicted_results):.3f}')
    print(f'Recall: {recall_score(test_labels, ensemble_predicted_results):.3f}')
    print(f'F1 Score: {f1_score(test_labels, ensemble_predicted_results):.3f}')
    print(f'Specificity: {recall_score(test_labels, ensemble_predicted_results, pos_label=0):.3f}')
    print(f'AUPRC: {average_precision_score(test_labels, ensemble_predicted_results):.3f}')
    print('')

    # Plot EnsembleBBB confusion matrix
    plot_confusion_matrix([tp, tn, fp, fn])


### Calcuate external testing results for B3clf ###
if args.b3clf:
    # Load in the B3clf training dataset, which has been created by removing molecules without 3D representations
    # from the B3DB dataset using minimize_with_rdkit()
    b3clf_train_dataset = PandasTools.LoadSDF('data/benchmark/train/b3clf_optimized_3d.sdf')
    invalid_3d_rep_count = 7807 - len(b3clf_train_dataset)
    
    # Identify charged molecules in B3clf train dataset
    charged_smiles_list = []
    for smiles in b3clf_train_dataset['ID']:
        mol = Chem.MolFromSmiles(smiles)

        # Get net formal charge of the Mol Object
        net_charge = Chem.GetFormalCharge(mol)

        if net_charge != 0:
            charged_smiles_list.append(smiles)

    # Filter the B3clf train dataset to remove charged molecules
    charged_smiles_count = len(charged_smiles_list)
    b3clf_train_dataset = b3clf_train_dataset[~b3clf_train_dataset['ID'].isin(charged_smiles_list)].copy()

    # Filter the B3clf testing results to exclude training data 
    overlapping_smiles_count = len(b3clf_results[b3clf_results['index'].isin(b3clf_train_dataset['ID'])])
    b3clf_results = b3clf_results[~b3clf_results['index'].isin(b3clf_train_dataset['ID'])].copy()

    # Merge B3clf prediction data and testing dataframe 
    b3clf_merged_dataframe = pd.merge(
        left=b3clf_results, 
        right=external_test_dataframe, 
        left_on='index', 
        right_on='SMILES', 
        how='left'
    )
    
    # Obtain prediction and true labels
    test_labels = b3clf_merged_dataframe['labels']
    b3clf_pred_labels = b3clf_merged_dataframe['B3clf_predicted_label']

    # Calculate TP, TN, FN, FP
    tn, fp, fn, tp = confusion_matrix(test_labels, b3clf_pred_labels).ravel()

    # Print B3clf testing metrics
    print('B3clf')
    print('')
    print(f'{invalid_3d_rep_count} compounds were removed from the B3clf training dataset due to invalid 3D representations')
    print(f'{charged_smiles_count} compounds were removed from the B3clf training dataset due to non-zero charges')
    print(f'{overlapping_smiles_count} compounds were removed from the DrugBank Database due to overlaps with the training dataset')
    print(f'{len(external_test_dataframe['SMILES'])-len(b3clf_results)} compounds were filtered from the DrugBank Database in total')
    print('')
    print(f'MCC Score: {matthews_corrcoef(test_labels, b3clf_pred_labels):.3f}')
    print(f'Accuracy: {accuracy_score(test_labels, b3clf_pred_labels):.3f}')
    print(f'Precision: {precision_score(test_labels, b3clf_pred_labels):.3f}')
    print(f'Recall: {recall_score(test_labels, b3clf_pred_labels):.3f}')
    print(f'F1 Score: {f1_score(test_labels, b3clf_pred_labels):.3f}')
    print(f'Specificity: {recall_score(test_labels, b3clf_pred_labels, pos_label=0):.3f}')
    print(f'AUPRC: {average_precision_score(test_labels, b3clf_pred_labels):.3f}')
    print('')

    # Plot B3clf confusion matrix
    plot_confusion_matrix(confusion_matrix_values=[tp, tn, fp, fn])


### Calculate external testing results for Deep-B3 ### 
if args.deep_b3:
    # Obtain Deep-B3 training dataset 
    deep_b3_train_dataset = pd.read_csv('data/benchmark/train/deep-b3_train_dataset.csv')
    
    # Filter Deep-B3 testing results to exclude training data
    overlapping_smiles_count = len(deep_b3_results[deep_b3_results['SMILES'].isin(deep_b3_train_dataset['smi'])])
    deep_b3_results = deep_b3_results[~deep_b3_results['SMILES'].isin(deep_b3_train_dataset['smi'])].copy()
    
    # Merge Deep-B3 and DrugBank testing dataframe
    deep_b3_merged_dataframe = pd.merge(
        left=external_test_dataframe, 
        right=deep_b3_results, 
        how='right', 
        left_on='SMILES', 
        right_on='SMILES'
    )
    
    # Obtain prediction and true labels
    deep_b3_predicted_labels = deep_b3_merged_dataframe['Permeable or Not']
    deep_b3_predicted_labels = [1 if deep_b3_predicted_labels[idx] == 'Permeable' else (0 if deep_b3_predicted_labels[idx] == 'Non-Permeable' else raise_err()) for idx in range(len(deep_b3_predicted_labels))] 
    test_labels = deep_b3_merged_dataframe['labels'].values
    
    # Calculate TP, TN, FN, FP
    tn, fp, fn, tp = confusion_matrix(test_labels, deep_b3_predicted_labels).ravel()
    
    # Print Deep-B3 testing metrics
    print('Deep-B3 Metrics:')
    print('')
    print(f'{overlapping_smiles_count} compounds were removed from the DrugBank Database due to overlap with the training dataset')
    print(f'{len(external_test_dataframe['SMILES'])-len(deep_b3_results)} compounds were filtered from the DrugBank Database in total')
    print('')
    print(f'MCC Score: {matthews_corrcoef(test_labels, deep_b3_predicted_labels):.3f}')
    print(f'Accuracy: {accuracy_score(test_labels, deep_b3_predicted_labels):.3f}')
    print(f'Precision: {precision_score(test_labels, deep_b3_predicted_labels):.3f}')
    print(f'Recall: {recall_score(test_labels, deep_b3_predicted_labels):.3f}')
    print(f'F1 Score: {f1_score(test_labels, deep_b3_predicted_labels):.3f}')
    print(f'Specificity: {recall_score(test_labels, deep_b3_predicted_labels, pos_label=0):.3f}')
    print(f'AUPRC: {average_precision_score(test_labels, deep_b3_predicted_labels):.3f}')
    print('')
    
    # Plot Deep-B3 confusion matrix
    plot_confusion_matrix([tp, tn, fp, fn])