from rdkit.Chem.SaltRemover import SaltRemover
from src.data_processing.dataset_fn import raw_bbb_data
from rdkit import Chem, RDLogger
import seaborn as sns, matplotlib.pyplot as plt, pandas as pd


# Import raw data without any curation methods
BBB_data = raw_bbb_data() 

# Prevent error messages from invalid SMILES configurations
RDLogger.DisableLog('rdApp.*')

# Add a new column in BBB_data for the mol object
mol_list = []
for smiles in BBB_data['SMILES']:
    mol_list.append(Chem.MolFromSmiles(smiles))
BBB_data['Mol'] = mol_list


### Check for Nonunique SMILES ### 
SMILES_count = BBB_data['SMILES'].value_counts()
nonunique_SMILES = SMILES_count[SMILES_count > 1]

print(f"Number of Nonunique SMILES: {len(nonunique_SMILES)}")
print(f"Total Number of Instances of Nonunique SMILES: {nonunique_SMILES.sum()}")
print(f"Total Number of Instances of Unique SMILES: {len(BBB_data['SMILES'])-nonunique_SMILES.sum()}")
print()

# Plot a Pie Chart representing the number of nonunique SMILES
explode = (0.1, 0)
plt.pie(labels=[f'Non-unique SMILES\n({nonunique_SMILES.sum()})', f'Unique SMILES\n({len(BBB_data['SMILES'])-nonunique_SMILES.sum()})'], 
        x=[nonunique_SMILES.sum(), len(BBB_data['SMILES'])-nonunique_SMILES.sum()],
        autopct='%1.1f%%',
        startangle=90, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}, 
        textprops={'fontsize': 14}, 
        shadow=True, 
        explode=explode
        ,colors=['#FF6188', '#AB9DF2'])
plt.legend(bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()


### Check for Inconsistently Reported Compounds ###
nonunique_SMILES_list = list(nonunique_SMILES.index)

# Define a final list to store all nonunique SMILES with conflicting BBclass values
curated_nonunique_SMILES_list = []

# Loop through each nonunique SMILES to find SMILES with conflicting BBclass values
for index in range(len(nonunique_SMILES_list)):
    num_classes = BBB_data.loc[BBB_data['SMILES'] == nonunique_SMILES_list[index]
                        ,'labels'].nunique()
    if num_classes > 1:
        curated_nonunique_SMILES_list.append(nonunique_SMILES_list[index])


### Check for Invalid SMILES ###
invalid_SMILES_count = 0
for mol in BBB_data['Mol']:
    if mol is None: 
        invalid_SMILES_count += 1
        continue
    else: 
        pass


### Check for Canonical SMILES ### 
canonical_smiles_count = 0
for mol, smiles in zip(BBB_data['Mol'], BBB_data['SMILES']): 
    if mol is not None:
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True) 
        if smiles == canonical_smiles: 
            canonical_smiles_count+= 1
        else: 
            pass 

print(f"Number of Canonical + Isomeric SMILES: {canonical_smiles_count}")
print(f"Number of Non Canonical + Isomeric SMILES: {len(BBB_data['SMILES'])-canonical_smiles_count}") 
print()

# Plot the number of canonical and non canonical SMILES on a pie chart
plt.pie(labels=[f'Canonical SMILES\n({canonical_smiles_count})', f'Noncanonical SMILES\n({len(BBB_data['SMILES'])-canonical_smiles_count})'], 
        x=[canonical_smiles_count, len(BBB_data['SMILES'])-canonical_smiles_count],
        autopct='%1.1f%%', 
        startangle=90, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}, 
        textprops={'fontsize': 14}, 
        shadow=True, 
        explode=explode,
        colors=['#FF6188', '#AB9DF2'])
plt.legend(bbox_to_anchor=(1, 0, 0.5, 1))
plt.tight_layout()
plt.show()


### Check for salts and solvents in SMILES ### 
remover = SaltRemover()
salt_count = 0
for mol in BBB_data['Mol']:
    if mol is not None: 
        stripped_mol, deleted_fragment = remover.StripMolWithDeleted(mol) 
        if deleted_fragment: 
            salt_count += 1 


### Check for a charge for each SMILES ### 
non_neutral = 0  
for mol in BBB_data['Mol']:
    if mol is not None: 
        charge = Chem.GetFormalCharge(mol)
        if charge != 0:
            non_neutral += 1 


### Check for explicit Hs for each SMILES ### 
explicit_hydrogen_count = 0 
for mol in BBB_data['Mol']:
    if mol is not None: 
        for atom in mol.GetAtoms(): 
            if atom.GetAtomicNum() == 1:
                explicit_hydrogen_count += 1
                break  


### Check for duplicate fragments for each SMILES ### 
duplicate_frag_count = 0 
for mol, smiles in zip(BBB_data['Mol'], BBB_data['SMILES']): 
    fragment = []
    if mol is not None: 
        fragments = smiles.split('.')
        for frag in fragments:   
            f_mol = Chem.MolFromSmiles(frag) 
            if f_mol is not None:
                f_mol = Chem.MolToSmiles(f_mol) 
                fragment.append(f_mol)
        unique_fragments = list(set(fragment))
        if len(unique_fragments) != len(fragment):
            duplicate_frag_count += 1 


### Check for SMILES with atomic numbers greater than 20 ### 
non_organic_count = 0 
allowed_atomic_nums = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
for mol in BBB_data['Mol']:
    if mol is not None: 
        for atom in mol.GetAtoms(): 
            if atom.GetAtomicNum() not in allowed_atomic_nums:
                non_organic_count += 1
                break  


### Plotting Function to Visualize SMILES Analysis ###
def plot_bar_graph(data: pd.DataFrame, 
                   property_count: int, 
                   name_with_property: str, 
                   name_without_property: str, 
                   palette: str):
    '''
    Plots a bar graph that contains the number of SMILES with the molecular propety and without 
    the molecular property given the formatted BBB dataset
    
    Args: 
        data: The BBB dataset to plot the bar graph for
        property_count: The number of SMILES with the molecular property
        name_with_property: The name for a SMILES with the molecular property
        name_without_property: The name for a SMILES without the molecular property
        palette: The specific palette color for seaborn's bar plot

    Returns: 
        A bar graph containing the number of SMILES with and without the molecular property
    ''' 

    # Define a set of labels containing the number of SMILES with and without the molecular property, and its percentage compared to the total SMILES count
    labels=[f"{property_count} ({(property_count/len(data['SMILES']))*100:.2f}%)", 
        f"{len(data['SMILES'])-property_count} ({((len(data['SMILES'])-property_count)/len(data['SMILES']))*100:.2f}%)"]
    
    # Plot the bar graph
    ax = sns.barplot(x=[name_with_property, name_without_property], 
                y=[property_count, len(data['SMILES'])-property_count], 
                edgecolor='black', 
                linewidth=1, 
                hue=[name_with_property, name_without_property], 
                palette=palette)
    
    # Label each bar according to the defined set of labels
    for idx, i in enumerate(ax.containers):
        if len(i.patches) > 0 and i.patches[0].get_height() > 0:
            ax.bar_label(i, 
                         labels=[labels[idx]],                              
                         padding=3)
            
    # Define the logarithmic y-scale and label
    ax.set_yscale('log')
    ax.set_ylabel('SMILES (Logarithmic)')

# Plot the calculated value of each molecular property on a bar graph
property_count_list = [len(curated_nonunique_SMILES_list), 
                       invalid_SMILES_count, 
                       salt_count, 
                       non_neutral, 
                       explicit_hydrogen_count, 
                       non_organic_count, 
                       duplicate_frag_count]

name_with_property_list = ['Inconsistently Reported SMILES', 
                           'Invalid SMILES', 
                           'SMILES with salts/solvents', 
                           'Non-Neutral SMILES', 
                           'SMILES with explicit Hs', 
                           'Non-Organic SMILES', 
                           'SMILES with duplicate frags']

name_without_property_list = ['Consistently Reported SMILES', 
                              'Valid SMILES', 
                              'SMILES without salts/solvents', 
                              'Neutral SMILES',
                              'SMILES without explicit Hs', 
                              'Organic SMILES', 
                              'SMILES without duplicate frags']

palette_list = ['colorblind', 
                'Pastel1', 
                'flare', 
                'coolwarm', 
                'magma', 
                'Set2', 
                'muted']

for idx in range(len(property_count_list)):
    print(f"Number of {name_with_property_list[idx]}: {property_count_list[idx]}")
    print(f"Number of {name_without_property_list[idx]}: {len(BBB_data['SMILES'])-property_count_list[idx]}")
    print()
    plot_bar_graph(BBB_data, 
                   property_count_list[idx], 
                   name_with_property_list[idx], 
                   name_without_property_list[idx], 
                   palette_list[idx])
    plt.show()