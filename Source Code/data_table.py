import pandas as pd 

# LightBBB DataSet
LightBBB = pd.read_csv('y_test_indices.csv')
LightBBB = LightBBB.rename(columns= {
    'Unnamed: 0': 'SMILES',
})
LightBBB_columns = LightBBB.columns

# MoleculeNet DataSet
MoleculeNet = pd.read_csv('MoleculeNet-BBBP-process-flow-step5-traindata.csv')
MoleculeNet = MoleculeNet.rename(columns={
    'BBB': 'BBclass'
})
MoleculeNet_columns = MoleculeNet.columns

# DeePred DataSet
DeePred = pd.read_csv('Table 1(Data Set).csv', encoding='cp1252', delimiter=',')
DeePred = DeePred.rename(columns= {
    'Compounds': 'SMILES',
    'BBB-Class': 'BBclass'
})
DeePred = DeePred.drop(['Unnamed: 5', 'Unnamed: 6'], axis=1)
DeePred_columns = DeePred.columns

# B3BD DataSet
B3BD = pd.read_csv('B3DB_regression.tsv', sep='\t')
B3BD_columns = B3BD.columns

# Convert logBB value into a binary representation of BBB permeability
logBB_value = []
for value in B3BD['logBB']: 
    if value >= 0.3: 
        logBB_value.append(1)
    elif value < 0.3: 
        logBB_value.append(0)
B3BD['BBclass'] = logBB_value