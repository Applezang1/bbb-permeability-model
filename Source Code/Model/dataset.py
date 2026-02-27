import pandas as pd

### Curate Dataset and Combine into one DataFrame ###
# Curate LightBBB Dataset
light_dataset = pd.read_csv('LightBBB.csv', usecols=['SMILES', 'labels'])

# Curate DeePred Datset
deepred_dataset = pd.read_csv('DeePred.csv', usecols=['SMILES', 'labels'])

# Curate MoleculeNet Dataset
molecule_dataset = pd.read_csv('MoleculeNet.csv', usecols=['SMILES', 'labels'])

# Curate B3DB Dataset
b3db_dataset = pd.read_csv('B3DB.tsv', usecols=['SMILES', 'logBB'], sep='\t')
b3db_bbb_class = []
for index in range(len(b3db_dataset['logBB'])):
    bbb_class = 1 if b3db_dataset['logBB'][index] > 0.3 else 0
    b3db_bbb_class.append(bbb_class)

b3db_dataset['labels'] = b3db_bbb_class
b3db_dataset = b3db_dataset.drop(columns='logBB')

# Concatenate the Datasets into one DataFrame
BBB_dataset = pd.concat([light_dataset, deepred_dataset, molecule_dataset, b3db_dataset]
                        ,ignore_index=True)


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



















    


