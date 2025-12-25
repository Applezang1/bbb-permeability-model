from data_analysis import * 
import numpy as np, matplotlib.pyplot as plt 

tpsa_values = [tpsa_positive, tpsa_negative]
logP_values = [logP_positive, logP_negative]

'''Make a Histogram of Molecular Weights for combined dataset'''
bins = np.arange(min(molecular_weights), max(molecular_weights) + 1, 50)
plt.hist(molecular_weights, edgecolor='black', align='mid', bins = bins)
ax = plt.gca()
ax.set_xlabel('Molecular Weight (amu)')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each Molecular Weight (amu) Category", size=11.5, weight='bold') 
plt.show()

'''Make a Histogram of LogP Value for combined dataset'''
bins = np.arange(min(logPs), max(logPs) + 1, 1)
plt.hist(logPs, edgecolor ='black', align='mid', bins = bins)
ax = plt.gca()
ax.set_xlabel('LogP Value')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each LogP Value Category", size=13.5, weight='bold') 
plt.show()

'''Make a Histogram of the TPSA Value for combined dataset'''
plt.hist(logPs, edgecolor ='black', align='mid', bins = 50)
ax = plt.gca()
ax.set_xlabel('TPSA (angstrom)')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each TPSA (angstrom) Category", size=11.5, weight='bold') 
plt.show()

'''Make a Box Plot of the TPSA Value, BBB+ and BBB-'''
fig, ax = plt.subplots(1, 2)
ax[0].boxplot(tpsa_positive, vert = False, showfliers = False)
ax[0].set_title("TPSA Distribution for BBB+ molecules", size=11.5, weight='bold')
ax[0].set_xlabel("TPSA (angstrom)")
ax[1].boxplot(tpsa_negative, vert = False, showfliers = False)
ax[1].set_title("TPSA Distribution for BBB- molecules", size=11.5, weight='bold')
ax[1].set_xlabel("TPSA (angstrom)")
plt.show()

'''Make a Box Plot of the logP Value, BBB+ and BBB-'''
fig, ax = plt.subplots(1, 2)
ax[0].boxplot(logP_positive, vert = False, showfliers = False)
ax[0].set_title("logP Distribution for BBB+ molecules", size=11.5, weight='bold')
ax[0].set_xlabel("logP")
ax[1].boxplot(logP_negative, vert = False, showfliers = False)
ax[1].set_title("logP Distribution for BBB- molecules", size=11.5, weight='bold')
ax[1].set_xlabel("logP")
plt.show()

'''Make a Histogram of LogP Value and TPSA for combined dataset'''
fig, ax = plt.subplots(1, 2)
bins = np.arange(min(logPs), max(logPs) + 1, 1)
ax[0].hist(logPs, edgecolor ='black', align='mid', bins = bins)
ax[0].set_xlabel('LogP Value')
ax[0].set_ylabel('Number of Molecules')
ax[0].set_title("Number of Molecules in each LogP Value Category", size=11.5, weight='bold') 
ax[1].hist(logPs, edgecolor ='black', align='mid', bins = 50)
ax[1].set_xlabel('TPSA (angstrom)')
ax[1].set_ylabel('Number of Molecules')
ax[1].set_title("Number of Molecules in each TPSA (angstrom) Category", size=11.5, weight='bold') 
plt.show()

'''Make a Histogram of LogP Value and TPSA each with BBB+ and BBB-'''
fig, ax = plt.subplots(2, 2)
ax[0, 0].hist(logP_positive, edgecolor ='black', align='mid', bins = 20)
ax[0, 0].set_xlabel('LogP Value')
ax[0, 0].set_ylabel('Number of Molecules')
ax[0, 0].set_title("LogP Distribution for BBB+ Molecules", size=11.5, weight='bold') 
ax[1, 0].hist(tpsa_positive, edgecolor ='black', align='mid', bins = 30)
ax[1, 0].set_xlabel('TPSA (angstrom)')
ax[1, 0].set_ylabel('Number of Molecules')
ax[1, 0].set_title("TPSA Distribution for BBB+ Molecules", size=11.5, weight='bold') 
ax[0, 1].hist(logP_negative, edgecolor ='black', align='mid', bins = 20)
ax[0, 1].set_xlabel('LogP Value')
ax[0, 1].set_ylabel('Number of Molecules')
ax[0, 1].set_title("LogP Distribution for BBB- Molecules", size=11.5, weight='bold') 
ax[1, 1].hist(tpsa_negative, edgecolor ='black', align='mid', bins = 30)
ax[1, 1].set_xlabel('TPSA (angstrom)')
ax[1, 1].set_ylabel('Number of Molecules')
ax[1, 1].set_title("TPSA Distribution for BBB- Molecules", size=11.5, weight='bold') 
plt.show()

'''Make a Vertical Box Plot of the TPSA and logP, BBB+ and BBB- with combined axis'''
fig, ax = plt.subplots(1, 2)
ax[0].boxplot(tpsa_values, vert = True, showfliers = False)
ax[0].set_xticks([1, 2], ['BBB+', 'BBB-'])
ax[0].set_title("TPSA Distribution for BBB+ and BBB-", size=11.5, weight='bold')
ax[0].set_ylabel("TPSA (angstrom)")
ax[0].set_ylim(-50, 500)
ax[1].boxplot(logP_values, vert = True, showfliers = False)
ax[1].set_xticks([1, 2], ['BBB+', 'BBB-'])
ax[1].set_title("logP Distribution for BBB+ and BBB-", size=11.5, weight='bold')
ax[1].set_ylabel("LogP Value")
ax[1].set_ylim(-5, 10)
plt.show()
