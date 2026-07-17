import pandas as pd, selfies as sf, numpy as np
from datasets import Dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


def raw_bbb_data(): 
    '''
    Combines the LightBBB, DeePred, MoleculeNet, and B3BD Dataset into one Pandas DataFrames

    Returns: 
        A 'BBB_dataset' containing each compound's SMILES and its respective BBB permeability label 
        before any curation methods
    '''

    # Curate LightBBB Dataset
    light_dataset = pd.read_csv('data/raw/LightBBB.csv', 
                                usecols=['SMILES', 'labels'])

    # Curate DeePred Datset
    deepred_dataset = pd.read_csv('data/raw/DeePred.csv', 
                                  usecols=['SMILES', 'labels'])

    # Curate MoleculeNet Dataset
    molecule_dataset = pd.read_csv('data/raw/MoleculeNet.csv', 
                                   usecols=['SMILES', 'labels'])

    # Curate B3DB Dataset
    b3db_dataset = pd.read_csv('data/raw/B3DB.tsv', 
                               usecols=['SMILES', 'logBB'],
                               sep='\t')
    
    dropped_indices = []
    b3db_bbb_class = []
    for index in range(len(b3db_dataset['logBB'])):
        if b3db_dataset['logBB'][index] > 0.3: 
            bbb_class = 1
            b3db_bbb_class.append(bbb_class)
        elif b3db_dataset['logBB'][index] < -1: 
            bbb_class = 0
            b3db_bbb_class.append(bbb_class)
        else: 
            dropped_indices.append(index)

    b3db_dataset = b3db_dataset.drop(index=dropped_indices).copy()
    b3db_dataset['labels'] = b3db_bbb_class
    b3db_dataset = b3db_dataset.drop(columns='logBB')

    # Concatenate the Datasets into one DataFrame
    BBB_dataset = pd.concat([light_dataset, deepred_dataset, molecule_dataset, b3db_dataset]
                            ,ignore_index=True)
    
    return BBB_dataset


def curate_bbb_data(BBB_dataset: pd.DataFrame): 
    '''
    Curates the BBB dataset from raw_bbb_data() with the following curation method: 
        1. Remove all Invalid SMILES from the dataset 
        2. Convert all SMILES to canoncicalized, isomeric SMILES 
        3. Remove salts and solvents from SMILES
        4. Neutralize non-neutral SMILES
        5. Remove explicit hydrogens in SMILES
        6. Remove duplicate fragments in SMILES
        7. Remove Duplicate SMILES with conflicting BBB Permeability Labels
        8. Keep one instance of each Duplicate SMILES
        9. Remove non-organic SMILES

    Args: 
        BBB_dataset: The Pandas DataFrame from raw_bbb_data()

    Returns: 
        A curated 'BBB_data' containing each compound's SMILES and its respective BBB permeability label
    '''

    # Prevent error messages from invalid SMILES configurations
    RDLogger.DisableLog('rdApp.*')
    
    ### 1. Curate the dataset to remove invalid SMILES ### 
    invalid_SMILES = []
    for smiles in BBB_dataset['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            invalid_SMILES.append(smiles) 
    
    BBB_data = BBB_dataset[~BBB_dataset['SMILES'].isin(invalid_SMILES)]

    ### 2. Canonicalize and convert SMILES to isomeric forms ###
    for index, smiles in enumerate(BBB_data['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        canonicalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        BBB_data.replace(smiles, canonicalized_smiles, inplace=True)

    # Add a new column in BBB_data for the mol object
    mol_list = []
    for smiles in BBB_data['SMILES']:
        mol_list.append(Chem.MolFromSmiles(smiles))
    BBB_data['Mol'] = mol_list

    ### 3. Check for salts and solvents in SMILES ### 
    remover = SaltRemover()
    for index in BBB_data.index:
        mol = BBB_data.at[index, 'Mol']
        stripped_mol, deleted_fragment = remover.StripMolWithDeleted(mol) 
        if deleted_fragment:
            stripped_smiles = Chem.MolToSmiles(stripped_mol, isomericSmiles=True)
            BBB_data.at[index, 'SMILES'] = stripped_smiles
            BBB_data.at[index, 'Mol'] = stripped_mol
    
    ### 4. Check for a charge for each SMILES ###
    uncharger = rdMolStandardize.Uncharger() 
    for index in BBB_data.index:
        mol = BBB_data.at[index, 'Mol']
        charge = Chem.GetFormalCharge(mol)
        if charge != 0:
            neutral_mol = uncharger.uncharge(mol)
            neutral_smiles = Chem.MolToSmiles(neutral_mol, isomericSmiles=True)
            BBB_data.at[index, 'SMILES'] = neutral_smiles
            BBB_data.at[index, 'Mol'] = neutral_mol
    
    ### 5. Check for explicit Hs for each SMILES ### 
    for index in BBB_data.index:
        mol = BBB_data.at[index, 'Mol']
        for atom in mol.GetAtoms(): 
            if atom.GetAtomicNum() == 1:
                inexplicit_Hs_mol = Chem.RemoveHs(mol)
                inexplicit_Hs_smiles = Chem.MolToSmiles(inexplicit_Hs_mol, isomericSmiles=True)
                BBB_data.at[index, 'Mol'] = inexplicit_Hs_mol
                BBB_data.at[index, 'SMILES'] = inexplicit_Hs_smiles
                break
    
    ### 6. Check for duplicate fragments for each SMILES ### 
    for index in BBB_data.index:
        mol = BBB_data.at[index, 'Mol']
        smiles = BBB_data.at[index, 'SMILES']
        fragments = smiles.split('.')
        unique_fragments = list(set(fragments))
        if len(unique_fragments) != len(fragments):
            clean_smiles = ".".join(unique_fragments)
            clean_mol = Chem.MolFromSmiles(clean_smiles) 
            BBB_data.at[index, 'SMILES'] = clean_smiles
            BBB_data.at[index, 'Mol'] = clean_mol
    
    ### 7/8. Curate the dataset based on repeated SMILES ###
    # Create a dataset of nonunique SMILES
    SMILES_count = BBB_data['SMILES'].value_counts()
    nonunique_SMILES = SMILES_count[SMILES_count > 1]

    # Define a list of all nonunique SMILES
    nonunique_SMILES_list = list(nonunique_SMILES.index)

    # Define a final list to store all nonunique SMILES with conflicting BBclass values
    curated_nonunique_SMILES_list = []

    # Loop through each nonunique SMILES to find SMILES with conflicting BBclass values
    for index in range(len(nonunique_SMILES_list)):
        num_classes = BBB_data.loc[BBB_data['SMILES'] == nonunique_SMILES_list[index]
                        ,'labels'].nunique()
        if num_classes > 1:
            curated_nonunique_SMILES_list.append(nonunique_SMILES_list[index])

    # Filter the data to not include final curated nonunique SMILES 
    BBB_data = BBB_data[~BBB_data['SMILES'].isin(curated_nonunique_SMILES_list)].copy()
    BBB_data.drop_duplicates(subset='SMILES', keep='first', inplace=True)
    BBB_data.reset_index(drop=True, inplace=True) 

    ### 9. Check for SMILES with atomic numbers greater than 20 ###
    inorganic_smiles = []
    allowed_atomic_nums = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    for mol in BBB_data['Mol']:
        for atom in mol.GetAtoms(): 
            if atom.GetAtomicNum() not in allowed_atomic_nums:
                inorganic_smiles.append(Chem.MolToSmiles(mol))
                break 
    
    BBB_data = BBB_data[~BBB_data['SMILES'].isin(inorganic_smiles)].copy()
    BBB_data = BBB_data.drop(columns='Mol')

    return BBB_data


def calculate_chem_features(BBB_data: pd.DataFrame):
    '''
    Calculates the logP, TPSA, molecular weight, NHOH, and NO count of each SMILES in the dataset. 

    Args: 
        BBB_data: The Pandas DataFrame from curate_bbb_data()

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
    for smiles in BBB_data['SMILES']:
        mol = Chem.MolFromSmiles(smiles) 
        logP.append(Descriptors.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))
        mol_weight.append(Descriptors.MolWt(mol))
        NHOH_count.append(rdMolDescriptors.CalcNumHBD(mol)) 
        NO_count.append(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8]))

    # Add feature values into the Pandas DataFrame
    BBB_data['logP'] = logP
    BBB_data['TPSA'] = tpsa
    BBB_data['Molecular Weight'] = mol_weight
    BBB_data['NHOH Count'] = NHOH_count
    BBB_data['NO Count'] = NO_count 

    return BBB_data


def convert_to_selfies(BBB_data: pd.DataFrame): 
    '''
    Converts the SMILES column in the dataset from curate_bbb_data() to its SELFIES representation

    Args: 
        BBB_data: The Pandas DataFrame from curate_bbb_data()

    Returns: 
        A 'BBB_data' with a SELFIES and BBB permeability label column
    '''
    
    # Initialize an empty selfies_list for storage
    selfies_list = []

    # Loop through the dataset and convert the SMILES to SELFIES
    for smiles in BBB_data['SMILES']: 
        selfies = sf.encoder(smiles) 
        selfies_list.append(selfies)
    
    # Update the SMILES column based on the computed SELFIES
    BBB_data['SELFIES'] = selfies_list
    BBB_data = BBB_data.drop(columns='SMILES')

    return BBB_data 


def augment_data(train_data: Dataset, 
                 num_augmentations: int, 
                 column_name: str, 
                 model_name: str):
    '''
    Augment the SMILES/SELFIES training dataset to include alternate representations of the same SMILES/SELFIES

    Args: 
        train_data: The training HuggingFace DataFrame originating from curate_bbb_data()
        num_augmentations: The number of times to perform data augmentation
        column_name: The column name representing the chemicals in the dataset (SMILES/SELFIES)
        model_name: The name of the model that will be trained by the training dataset

    Returns: 
        A training HuggingFace DataFrame with augmented data
    ''' 

    # Prevent error messages from invalid SMILES configurations
    RDLogger.DisableLog('rdApp.*')

    # Convert the inputted HuggingFace DataFrame into a Pandas DataFrame for compatability
    train_data = train_data.to_pandas()
    rng = np.random.default_rng() 

    # Initialize lists for storage
    augmented_list = []
    labels_list = []

    if column_name == 'SMILES':
        # Loop through the dataset a set number of times to augment the SMILES data 
        for i in range(num_augmentations):
            for smiles, label in zip(train_data[column_name], train_data['labels']):
                # Augment the SMILES string
                mol = Chem.MolFromSmiles(smiles)
                new_order = list(range(mol.GetNumAtoms()))
                rng.shuffle(new_order)
                new_mol = Chem.RenumberAtoms(mol,new_order)
                new_smiles = Chem.MolToSmiles(new_mol, canonical=False)

                # Store the smiles and labels into respective lists
                augmented_list.append(new_smiles)
                labels_list.append(label)

    elif column_name == 'SELFIES':
        # Loop through the dataset a set number of times to augment the SELFIES data
        for i in range(num_augmentations):
            for selfies, label in zip(train_data[column_name], train_data['labels']):
                # Convert the SELFIES back to SMILES
                smiles = sf.decoder(selfies)

                # Augment the SMILES string
                mol = Chem.MolFromSmiles(smiles)
                new_order = list(range(mol.GetNumAtoms()))
                rng.shuffle(new_order)
                new_mol = Chem.RenumberAtoms(mol,new_order)
                new_smiles = Chem.MolToSmiles(new_mol, canonical=False)

                # Convert the augmented SMILES back to SELFIES
                new_selfies = sf.encoder(new_smiles)

                # Store selfies and label into respective lists
                augmented_list.append(new_selfies)
                labels_list.append(label)

    # Store the augmented data into a dictionary
    augmented_data = {
        column_name: augmented_list,
        'labels': labels_list
    }

    # Create a new Pandas DataFrame with the augmented data
    augmented_BBB_data = pd.DataFrame(augmented_data)

    # Implement required spacing to SELFIES strings if the model name is SELFIES-TED
    if model_name == 'SELFIES-TED':
        augmented_BBB_data = space_selfies_strings(augmented_BBB_data)


    return pd.concat([train_data, augmented_BBB_data], ignore_index=True)


def space_selfies_strings(data: pd.DataFrame):
    '''
    Spaces all the SELFIES tokens for each SELFIES in the SELFIES column of the inputted DataFrame

    Args: 
        data: The Pandas DataFrame to space the SELFIES tokens on

    Returns: 
        A Pandas DataFrame with spaced SELFIES strings
    ''' 
    
    # Initialize list to store spaced selfies strings
    spaced_selfies = []

    # Loop through the SELFIES column to space SELFIES tokens
    for selfies in data['SELFIES']: 
        new_selfies = selfies.replace("][", "] [")
        spaced_selfies.append(new_selfies)

    # Create a new Pandas DataFrame with spaced SELFIES strings
    data['SELFIES'] = spaced_selfies 

    return data














    


