# Import Functions and Files
from data_table import LightBBB, MoleculeNet, DeePred, B3BD
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import numpy as np, matplotlib.pyplot as plt 
import pandas as pd 
from scipy.stats import ttest_ind
from scipy import stats

# Variables
BBB_permeable_number = 0
BBB_nonpermeable_number = 0
invalid_molecule = []
molecular_weights = []
logPs = []
TPSAs = []
BBB_permeable = []
BBB_nonpermeable = []
unique_multiple_BBB_compound = []
invalid_multiple_BBB_compound = []

'''Find Number of SMILES Duplicate'''
data_table = pd.concat([LightBBB[['SMILES','BBclass']], MoleculeNet[['SMILES','BBclass']], DeePred[['SMILES','BBclass']], 
                        B3BD[['SMILES','BBclass']]], ignore_index=True) # Data Table with SMILES, BBclass
SMILES_count = data_table['SMILES'].value_counts() # Count the amount of times each SMILES occurs
duplicates = SMILES_count[SMILES_count > 1] # 1463 duplicate SMILES

'''Count the Number of Unique Compounds'''  
unique_compounds = SMILES_count[SMILES_count < 2] # 10635 unique SMILES

'''Print the Number of SMILES Duplicate and Unique SMILES'''
print(f"There are a total of {duplicates} SMILES in the data table") 
print(f"There are a total of {unique_compounds} SMILES in the data table")

'''Table of Duplicate Compounds and Reported BBclass Data'''
duplicate_SMILES = duplicates.index.to_numpy() # Makes a numpy array of all the duplicate SMILES
duplicate_SMILES_table = data_table[data_table['SMILES'].isin(duplicate_SMILES)] # Makes a table of every instance of duplicate SMILES 
duplicate_SMILES_table = duplicate_SMILES_table.groupby('SMILES')['BBclass'].nunique() # Makes a table that shows the number of unique reported values

'''Filtering Duplicates based on whether the reported values are consistent or not'''
for i in range(len(duplicate_SMILES_table)):
    if duplicate_SMILES_table.iloc[i] == 1: 
        unique_multiple_BBB_compound.append(duplicate_SMILES_table.index[i])
    else: 
        invalid_multiple_BBB_compound.append(duplicate_SMILES_table.index[i])
unique_SMILES_table = data_table[~data_table['SMILES'].isin(invalid_multiple_BBB_compound)]

'''Find Molecular Weight and remove invalid SMILES'''
for smiles in unique_SMILES_table['SMILES']:
    molecule = Chem.MolFromSmiles(smiles) # Convert SMILES to molecular name
    if molecule: 
        molecular_weight = Descriptors.MolWt(molecule) # Find the molecular weight for each SMILES
        molecular_weights.append(molecular_weight) # Add molecular weight to list
    else:
        invalid_molecule.append(smiles) # Add invalid SMILES to invalid_molecule  
unique_SMILES_table = unique_SMILES_table[~unique_SMILES_table['SMILES'].isin(invalid_molecule)] # Keep all SMILES that are valid
unique_SMILES_table['Molecular Weight (amu)'] = molecular_weights # Add molecular weights to Data Table

'''Find logP value and remove invalid SMILES'''
for smiles in unique_SMILES_table['SMILES']:
    molecule = Chem.MolFromSmiles(smiles) # Convert SMILES to molecular name 
    logP = Crippen.MolLogP(molecule) # Find the logP value for each SMILES 
    logPs.append(logP) 
unique_SMILES_table['LogP Value'] = logPs # Add logP value to the Data Table

'''Find TPSA value and remove invalid SMILES'''
for smiles in unique_SMILES_table['SMILES']:
    molecule = Chem.MolFromSmiles(smiles) # Convert SMILES to molecular name 
    TPSA = rdMolDescriptors.CalcTPSA(molecule) # Find the TPSA value for each SMILES 
    TPSAs.append(TPSA) 
unique_SMILES_table['TPSA Value'] = TPSAs # Add TPSA value to the Data Table

'''Find # of BBB permeable and nonpermeable Drugs'''
for value in unique_SMILES_table['BBclass']: 
    if value == 1: 
        BBB_permeable_number = BBB_permeable_number + 1
    elif value == 0: 
        BBB_nonpermeable_number = BBB_nonpermeable_number + 1
print(f'The number of molecules in the data table that are BBB permeable is {BBB_permeable_number}')  # 9589 BBB+ molecules
print(f'The number of molecules in the data table that are BBB nonpermeable is {BBB_nonpermeable_number}') # 3601 BBB- molecules

'''Distribute SMILES into separate tables based on BBB permeability'''
BBB_permeable = unique_SMILES_table[unique_SMILES_table['BBclass'] == 1]['SMILES'].tolist()
BBB_nonpermeable = unique_SMILES_table[unique_SMILES_table['BBclass'] == 0]['SMILES'].tolist()
BBB_permeable_table = unique_SMILES_table[~unique_SMILES_table['SMILES'].isin(BBB_nonpermeable)] # Keep all BBB+ permeable molecules
BBB_nonpermeable_table = unique_SMILES_table[~unique_SMILES_table['SMILES'].isin(BBB_permeable)] # Keep all BBB+ nonpermeable molecules

'''Get TPSA Values for BBB+ and BBB- molecules individually'''
tpsa_positive = BBB_permeable_table['TPSA Value']
tpsa_negative = BBB_nonpermeable_table['TPSA Value']

'''Get logP Values for BBB+ and BBB- molecules individually'''
logP_positive = BBB_permeable_table['LogP Value']
logP_negative = BBB_nonpermeable_table['LogP Value']





