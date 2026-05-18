from src.data_processing.dataset_fn import raw_bbb_data, curate_bbb_data, calculate_chem_features
import numpy as np, matplotlib.pyplot as plt
import seaborn as sns

### Initialize All Required Datasets for Data Analysis ###
# Create Pandas DataFrame of SMILES and BBB permeability label
BBB_data = curate_bbb_data(raw_bbb_data())

# Calculate the logP, TPSA, molecular weight, NHOH, NO count
BBB_data = calculate_chem_features(BBB_data)


### Correlation Matrixes ###
# Correlation Matrix between molecular properties
correlation_matrix = BBB_data[['logP', 'TPSA', 'Molecular Weight', 'NO Count', 'NHOH Count']].corr(numeric_only=True)
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm')
plt.show()

# Correlation Matrix between molecular property and compound label
molecular_property = 'logP'
correlation_matrix = BBB_data[[molecular_property, 'labels']].corr(numeric_only=True)
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm')
plt.show()


### Class Imbalance Check ###
sns.barplot(x=['BBB+', 'BBB-'], 
            y=[(BBB_data['labels'] == 1).sum(), (BBB_data['labels'] == 0).sum()],
            edgecolor='black', 
            linewidth=1)
plt.ylabel('Number of Compounds')
plt.show() 

# Print the number of BBB+ and BBB- compounds
print(f"BBB+ Count: {(BBB_data['labels'] == 1).sum()}")
print(f"BBB- Count: {(BBB_data['labels'] == 0).sum()}")


### Generate Violin Plots to show the distribution of each molecular property for BBB+ and BBB- class ###
# Violin Plot of logP for BBB+ and BBB-
label_names = {0: 'BBB-', 1: 'BBB+'}
plt.subplot(2, 2, 1)
sns.violinplot(data=BBB_data, 
               x='logP', 
               y='labels', 
               hue='labels', 
               orient='h', 
               formatter=label_names, 
               legend=False)
plt.xlabel('logP')
plt.title('LogP Distribution', fontsize=10)

# Violin Plot of TPSA for BBB+ and BBB-
plt.subplot(2, 2, 2)
sns.violinplot(data=BBB_data, 
               x='TPSA', 
               y='labels', 
               hue='labels', 
               orient='h', 
               formatter=label_names, 
               legend=False)
plt.xlabel('TPSA')
plt.title('TPSA Distribution', fontsize=10)

# Violin Plot of Molecular Weight for BBB+ and BBB-
plt.subplot(2, 2, 3)
sns.violinplot(data=BBB_data, 
               x='Molecular Weight', 
               y='labels', 
               hue='labels', 
               orient='h', 
               formatter=label_names, 
               legend=False)
plt.xlabel('Molecular Weight')
plt.title('Molecular Weight Distribution', fontsize=10)

# Violin Plot of NHOH Count for BBB+ and BBB-
plt.subplot(2, 2, 4)
sns.violinplot(data=BBB_data, 
               x='NHOH Count', 
               y='labels', 
               hue='labels', 
               orient='h', 
               formatter=label_names, 
               legend=False)
plt.xlabel('NHOH Count')
plt.title('NHOH Distribution', fontsize=10)

plt.tight_layout()
plt.show()


### Outlier Detection for Molecular Properties ###
def outlier_detection(data, label, property): 
    '''
    Uses the 1.5 IQR Rule to calculate outliers for the formatted BBB dataset
    
    Args: 
        data: The BBB dataset in which the outlier will be calculated in
        label: The class label for outlier detection 
            - 0: BBB- 
            - 1: BBB+
        property: The molecular property that the outlier detection will calculate for outliers

    Returns: 
        A printed statement of numerical ranges that are classified as outliers
    ''' 

    q1 = np.percentile(data.loc[data['labels']==label, property], 25)
    q3 = np.percentile(data.loc[data['labels']==label, property], 75)
    iqr = q3 - q1
    if label == 1:
      print(f"BBB+ {property} Outlier: Anything below {(q1 - 1.5*iqr):.3f}" 
            f" and above {(q3 + 1.5*iqr):.3f} are outliers")
    else:
      print(f"BBB- {property} Outlier: Anything below {(q1 - 1.5*iqr):.3f}"
            f" and above {(q3 + 1.5*iqr):.3f} are outliers")

# Determine the outlier ranges for logP, TPSA, Molecular Weight, and NHOH Count
properties_to_check = ['logP', 'TPSA', 'Molecular Weight', 'NHOH Count']
for property in properties_to_check:
      outlier_detection(BBB_data, 1, property)
      outlier_detection(BBB_data, 0, property)
      print()

