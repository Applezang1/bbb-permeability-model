import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

def raw_bbb_data(): 
    '''
    Combines the LightBBB, DeePred, MoleculeNet, and B3BD Dataset into one Pandas DataFrames

    Returns: 
        A 'BBB_dataset' containing each compound's SMILES and its respective BBB permeability label 
        before any curation methods
    '''
    # Curate LightBBB Dataset
    light_dataset = pd.read_csv('data/raw/LightBBB.csv', usecols=['SMILES', 'labels'])

    # Curate DeePred Datset
    deepred_dataset = pd.read_csv('data/raw/DeePred.csv', usecols=['SMILES', 'labels'])

    # Curate MoleculeNet Dataset
    molecule_dataset = pd.read_csv('data/raw/MoleculeNet.csv', usecols=['SMILES', 'labels'])

    # Curate B3DB Dataset
    b3db_dataset = pd.read_csv('data/raw/B3DB.tsv', usecols=['SMILES', 'logBB'], sep='\t')
    b3db_bbb_class = []
    for index in range(len(b3db_dataset['logBB'])):
        bbb_class = 1 if b3db_dataset['logBB'][index] > 0.3 else 0
        b3db_bbb_class.append(bbb_class)

    b3db_dataset['labels'] = b3db_bbb_class
    b3db_dataset = b3db_dataset.drop(columns='logBB')

    # Concatenate the Datasets into one DataFrame
    BBB_dataset = pd.concat([light_dataset, deepred_dataset, molecule_dataset, b3db_dataset]
                            ,ignore_index=True)
    
    return BBB_dataset


def curate_bbb_data(BBB_dataset): 
    '''
    Curates the BBB dataset from raw_bbb_data() with the following curation method: 
        1. Remove all Invalid SMILES from the dataset 
        2. Convert all SMILES to canoncicalized, isomeric SMILES 
        3. Remove Duplicate SMILES with conflicting BBB Permeability Labels
        4. Keep one instance of each Duplicate SMILES

    Args: 
        BBB_dataset: The Pandas DataFrame from raw_bbb_data()

    Returns: 
        A curated 'BBB_data' containing each compound's SMILES and its respective BBB permeability label
    '''

    # Prevent error messages from invalid SMILES configurations
    RDLogger.DisableLog('rdApp.*')
    
    ### Curate the dataset to remove invalid SMILES ### 
    invalid_SMILES = []
    for smiles in BBB_dataset['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            invalid_SMILES.append(smiles) 

    BBB_dataset = BBB_dataset[~BBB_dataset['SMILES'].isin(invalid_SMILES)]

    ### Canonicalize and convert SMILES to isomeric forms ###
    for index, smiles in enumerate(BBB_dataset['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        canonicalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        BBB_dataset.replace(smiles, canonicalized_smiles, inplace=True)
    
    ### Curate the dataset based on repeated SMILES ###
    # Create a dataset of nonunique SMILES
    SMILES_count = BBB_dataset['SMILES'].value_counts()
    nonunique_SMILES = SMILES_count[SMILES_count > 1]

    # Define a list of all nonunique SMILES
    nonunique_SMILES_list = list(nonunique_SMILES.index)

    # Define a final list to store all nonunique SMILES with conflicting BBclass values
    curated_nonunique_SMILES_list = []

    # Loop through each nonunique SMILES to find SMILES with conflicting BBclass values
    for index in range(len(nonunique_SMILES_list)):
        num_classes = BBB_dataset.loc[BBB_dataset['SMILES'] == nonunique_SMILES_list[index]
                        ,'labels'].nunique()
        if num_classes > 1:
            curated_nonunique_SMILES_list.append(nonunique_SMILES_list[index])

    # Filter the data to not include final curated nonunique SMILES 
    BBB_data = BBB_dataset[~BBB_dataset['SMILES'].isin(curated_nonunique_SMILES_list)].copy()
    BBB_data.drop_duplicates(subset='SMILES', keep='first', inplace=True)
    BBB_data.reset_index(drop=True, inplace=True) 

    return BBB_data


def calculate_chem_features(data):
    '''
    Calculates the logP, TPSA, molecular weight, NHOH, and NO count of each SMILES in the dataset. 
    Additionally filters out invalid SMILES strings from the inputted data

    Args: 
        data: The Pandas DataFrame from curate_bbb_data()

    Returns: 
        A 'BBB_data' containing the logP, TPSA, molecular weight, NHOH, and NO count of each SMILES
    '''
    
    # Initialize empty lists to store molecular features and invalid SMILES
    logP = []
    tpsa = []
    mol_weight = []
    NHOH_count = [] 
    NO_count = []

    # Prevent error messages from invalid SMILES configurations
    RDLogger.DisableLog('rdApp.*')

    # Iterate through the dataset to calculate the logP, TPSA, molecular weight, NHOH, and NO count 
    for smiles in data['SMILES']:
        mol = Chem.MolFromSmiles(smiles) 
        logP.append(Descriptors.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))
        mol_weight.append(Descriptors.MolWt(mol))
        NHOH_count.append(rdMolDescriptors.CalcNumHBD(mol)) 
        NO_count.append(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8]))

    # Add feature values into the Pandas DataFrame
    data['logP'] = logP
    data['TPSA'] = tpsa
    data['Molecular Weight'] = mol_weight
    data['NHOH Count'] = NHOH_count
    data['NO Count'] = NO_count 

    return data



















    


