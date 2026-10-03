import pandas as pd, umap, torch, matplotlib.pyplot as plt, seaborn as sns
from src.data_processing.dataset_fn import curate_bbb_data, raw_bbb_train_data, raw_bbb_external_test_data, raw_bbb_test_data, remove_train_test_conflicts
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
from src.data_processing.dataloaders import create_dataloader, tokenize_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import load_model
from datasets import Dataset


# Define training, testing, and external testing datasets 
test_dataset = curate_bbb_data(raw_bbb_test_data())
external_test_dataset = curate_bbb_data(raw_bbb_external_test_data())
train_dataset = curate_bbb_data(raw_bbb_train_data())
train_dataset, test_dataset = remove_train_test_conflicts(train_dataset, test_dataset)
train_dataset, external_test_dataset = remove_train_test_conflicts(train_dataset, external_test_dataset)

# Import dataset demonstrating the database contributions for all duplicate SMILES instances
duplicate_dataset = pd.read_csv('data/database_contribution_data.csv')

# Add a new column in duplicate_dataset for the mol object
mol_list = []
for smiles in duplicate_dataset['SMILES']:
    mol_list.append(Chem.MolFromSmiles(smiles))
duplicate_dataset['Mol'] = mol_list

# Remove inorganic SMILES from duplicate_dataset
inorganic_smiles = []
allowed_atomic_nums = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
for mol in duplicate_dataset['Mol']:
    for atom in mol.GetAtoms(): 
        if atom.GetAtomicNum() not in allowed_atomic_nums:
            inorganic_smiles.append(Chem.MolToSmiles(mol))
            break 
    
duplicate_dataset = duplicate_dataset[~duplicate_dataset['SMILES'].isin(inorganic_smiles)].copy()
duplicate_dataset = duplicate_dataset.drop(columns='Mol')

# Define a nonunique_SMILES dataset demonstrating the value count of each duplicate SMILES
SMILES_count = duplicate_dataset['SMILES'].value_counts()
nonunique_SMILES = SMILES_count[SMILES_count > 1]

# Define a list of all nonunique SMILES
nonunique_SMILES_list = list(nonunique_SMILES.index)


### Supplementary Figure 1B and 1C Calculation ###
# Initialize variables for data storage
light, deepred, b3db, molecule = 0, 0, 0, 0
light_BBB_positive, deepred_BBB_positive, b3db_BBB_positive, molecule_BBB_positive = 0, 0, 0, 0
light_BBB_negative, deepred_BBB_negative, b3db_BBB_negative, molecule_BBB_negative = 0, 0, 0, 0

# Define a loop that calculates the contributions (Overall Dataset Number, BBB+, BBB-) of each database toward a duplicate SMILES using the following formula 
# If there is an instance of a duplicate SMILES from a certain database, divide 1 over the number duplicate SMILES instances to add towards the database contribution
for smiles in nonunique_SMILES_list:
    duplicate_smiles_instances = duplicate_dataset.loc[duplicate_dataset['SMILES'] == smiles] 
    for idx in duplicate_smiles_instances['Source']:
        if idx == 'DeePred':
            deepred += 1/len(duplicate_smiles_instances)
            deepred_exist = True
        elif idx == 'B3DB':
            b3db += 1/len(duplicate_smiles_instances)
            b3db_exist=True
        elif idx == 'MoleculeNet':
            molecule += 1/len(duplicate_smiles_instances)
            molecule_exist=True
        elif idx == 'LightBBB':
            light += 1/len(duplicate_smiles_instances)
            light_exist=True
    for labels in duplicate_smiles_instances['labels']:
        label = labels
    if label == 0:
        if light_exist == True:
            light_BBB_negative += 1/len(duplicate_smiles_instances)
        if deepred_exist == True:
            deepred_BBB_negative += 1/len(duplicate_smiles_instances)
        if b3db_exist == True:
            b3db_BBB_negative += 1/len(duplicate_smiles_instances) 
        if molecule_exist == True:
            molecule_BBB_negative += 1/len(duplicate_smiles_instances)
    elif label == 1:
        if light_exist == True:
            light_BBB_positive += 1/len(duplicate_smiles_instances)
        if deepred_exist == True:
            deepred_BBB_positive += 1/len(duplicate_smiles_instances)
        if b3db_exist == True:
            b3db_BBB_positive += 1/len(duplicate_smiles_instances) 
        if molecule_exist == True:
            molecule_BBB_positive += 1/len(duplicate_smiles_instances)
    deepred_exist, b3db_exist, molecule_exist, light_exist = False, False, False, False

# Remove nonunique SMILES from training dataset
train_dataset = train_dataset[~train_dataset['SMILES'].isin(nonunique_SMILES_list)].copy()
print(train_dataset)

# Give dataset contributions (Overall Dataset Number, BBB+, BBB-) for nonunique training SMILES dataset
for row, label in zip(train_dataset['Source'], train_dataset['labels']):
    if row == 'DeePred':
        deepred += 1
        if label == 0:
            deepred_BBB_negative += 1
        elif label == 1:
            deepred_BBB_positive += 1
    elif row == 'B3DB':
        b3db += 1
        if label == 0: 
            b3db_BBB_negative += 1
        elif label == 1:
            b3db_BBB_positive += 1
    elif row == 'MoleculeNet':
        molecule += 1
        if label == 0:
            molecule_BBB_negative += 1
        elif label == 1:
            molecule_BBB_positive += 1
    elif row == 'LightBBB':
        light += 1
        if label == 0:
            light_BBB_negative += 1
        elif label == 1:
            light_BBB_positive += 1

# Print results
print(f'LightBBB Dataset Contribution: {light}')
print(f'MoleculeNet Dataset Contribution: {molecule}')
print(f'B3DB Dataset Contribution {b3db}')
print(f'DeePred-BBB Dataset Contribution {deepred}')
print('\n')
print('Positives (BBB+) Contribution')
print(f'B3DB Dataset BBB+ Count: {b3db_BBB_positive}')
print(f'MoleculeNet Dataset BBB+ Count: {molecule_BBB_positive}')
print(f'LightBBB Dataset BBB+ Count: {light_BBB_positive}')
print(f'DeePred-BBB Dataset BBB+ Count: {deepred_BBB_positive}')
print('\n')
print('Negatives (BBB-) Contribution')
print(f'B3DB Dataset BBB- Count: {b3db_BBB_negative}')
print(f'MoleculeNet Dataset BBB- Count: {molecule_BBB_negative}')
print(f'LightBBB Dataset BBB- Count: {light_BBB_negative}')
print(f'DeePred-BBB Dataset BBB- Count: {deepred_BBB_negative}')


### Supplementary Figure 1A Calculation ###
# Define database containing conflicting SMILES with database source
conflict_database = pd.read_csv('data/conflicting_smiles_data.csv')

# Initialize variables for value storage
light_bbb, deepred_bbb, molecule_net, b3db = 0, 0, 0, 0

# Define a loop that goes through the conflict_database to count the contributing databases
for publication in conflict_database['Publication']:
    if publication == 'LightBBB':
        light_bbb += 1
    elif publication == 'DeePred':
        deepred_bbb += 1
    elif publication == 'MoleculeNet':
        molecule_net += 1
    elif publication == 'B3DB':
        b3db += 1

print('\n')
print('Conflicting SMILES for each database')
print(f'LightBBB Conflicting SMILES Count: {light_bbb}')
print(f'DeePred-BBB Conflicting SMILES Count: {deepred_bbb}')
print(f'MoleculeNet Conflicting SMILES Count: {molecule_net}')
print(f'B3DB Conflicting SMILES Count: {b3db}\n')


### Supplementary Figure 3B Calculations ###
# Define training dataset
train_dataset = pd.read_csv('data/processed/smiles_train_dataframe.csv')

# Define list to store all scaffold SMILES
scaffold_smiles_list = []

# Define a loop that generates the scaffold SMILES for each SMILES in the train dataset
for smiles in train_dataset['SMILES']:
    scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    scaffold_smiles_list.append(scaffold_smiles)

# Return the top 20 most common scaffolds
counts = Counter(scaffold_smiles_list)
top_twenty = counts.most_common(20)
print(top_twenty)


### Supplementary Figure 3A Calculations ###
# Define train_val, testing, and external testing datasets 
test_dataset = curate_bbb_data(raw_bbb_test_data())
external_test_dataset = curate_bbb_data(raw_bbb_external_test_data())
train_dataframe = curate_bbb_data(raw_bbb_train_data())
train_dataframe, test_dataset = remove_train_test_conflicts(train_dataframe, test_dataset)
train_dataframe, external_test_dataset = remove_train_test_conflicts(train_dataframe, external_test_dataset)

# Define list to store DataBase labels
database_publication_list = train_dataframe['Source'] 

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                                  trust_remote_code=True)

# Convert the train and validation Pandas DataFrames to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_dataframe)

# Tokenize training dataset         
train_dataset = tokenize_dataset(train_dataset,
                                 tokenizer, 
                                 'SMILES')
    

# Create PyTorch training dataloader for input into MoLFormer    
train_dataloader = create_dataloader(train_dataset, 
                                     batch_size=32, 
                                     shuffle=False,
                                     num_workers=0)

# Create device-agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load in trained model
model = AutoModelForSequenceClassification.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                                            deterministic_eval=True, 
                                                            trust_remote_code=True, 
                                                            num_labels=2, 
                                                            use_safetensors=True, 
                                                            classifier_dropout_prob=0.4970983440156713)
saved_model = load_model(model, f'saved_models/MolFormer-XL-0.pth')
saved_model.to(device)

# Input training dataloader into MoLFormer model
saved_model.eval() 

# Define special_ids tokens in the MoLFormer tokenizer
special_ids = torch.tensor(tokenizer.all_special_ids, device=device)

# Initialize empty list to store compound vectors
compound_vec_list = []

# Run validation loop for each batch in the training dataloader to obtain contextualized embeddings for all hidden layers
with torch.inference_mode():
    for batch, input in enumerate(train_dataloader): 
        # Put data onto target device
        input = {k: v.to(device) for k, v in input.items()} 

        # Compute a forward pass 
        output = model(**input, output_hidden_states=True) 

        # Define a boolean mask for chemical/content tokens
        is_special_mask = torch.isin(input["input_ids"], special_ids)
        masking_boolean = ~is_special_mask

        # Access the last four hidden layer embeddings 
        all_hidden_states = output.hidden_states
        last_four_hidden_embeddings = all_hidden_states[-4:]

        # Perform layer aggregation on hidden embeddings
        aggregated_hidden_embeddings = sum(last_four_hidden_embeddings)

        # Apply avg, cls, and avg-ns pooling method
        for seq in range(len(aggregated_hidden_embeddings)):
            # Perform cls pooling method
            cls_pool = aggregated_hidden_embeddings[seq, 0, :]

            # Perform avg pooling method (without considering padding)
            seq_pad_mask = input["attention_mask"][seq] == 0
            seq_pad_mask = ~seq_pad_mask
            avg_pool = aggregated_hidden_embeddings[seq, :, :]
            
            avg_pool = avg_pool[seq_pad_mask]
            avg_pool = torch.mean(avg_pool, dim=0)

            # Perform avg-ns pooling method 
            avg_ns_pool = aggregated_hidden_embeddings[seq, :, :]
            avg_ns_pool = avg_ns_pool[masking_boolean[seq, :]]
            avg_ns_pool = torch.mean(avg_ns_pool, dim=0)

            # Concatenate results into one vector 
            compound_vec_concat = torch.cat((cls_pool, avg_pool, avg_ns_pool), dim=0).cpu()
            compound_vec_list.append(compound_vec_concat)


# Define UMAP-dimensionality reducer 
reducer = umap.UMAP(
    n_neighbors = 15, 
    min_dist = 0.1, 
    metric = 'cosine', 
    random_state= 42
)

# Create UMAP object
umap_2d = reducer.fit_transform(compound_vec_list)

# Create dataframe containing UMAP dimensions
umap_df = pd.DataFrame(umap_2d, columns=['UMAP 1', 'UMAP 2'])
umap_df['Label'] = database_publication_list

# Plot UMAP figure
plt.figure(figsize=(8, 6))
sns.scatterplot(data=umap_df, 
                x='UMAP 1', 
                y='UMAP 2', 
                hue="Label",)
plt.show()







    

