# Paper 2: Blood-brain barrier penetration prediction enhanced by uncertainty estimation

<ins>Paper 2 Link</ins>: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00619-2?utm_source=chatgpt.com  

## Key Terminology for Contextual Understanding: 

<ins>BBBp</ins>: a quantifiable measurement of the blood brain barrier’s permeability to a specific molecule

<ins>Uncertainty estimation</ins>: a quantifiable value of the model’s accuracy of a prediction or output

<ins>PCP</ins>: a molecule’s chemical and physical characteristics and attributes 

<ins>Molecular fingerprint</ins>: a digital representation of a molecule’s structural features such as atoms and the bonds between them

<ins>GNN (Graph Neural Network)</ins>: a deep learning model that transforms the nodes (vertices/object) and edges (relationship between nodes) of a graph into quantifiable data or numerical representations

<ins>Transformer-based architecture</ins>: a type of deep learning model that calculates the relationships between the inputted data

<ins>Attentive FP</ins>: a type of GNN that learns molecular properties through data and graphs such as molecular fingerprints (where nodes: atoms, edges: bonds)

<ins>GROVER</ins>: a type of GNN that uses attention mechanisms to calculate general molecular embeddings (numerical vector representation of a molecule including data such as atoms, bonds, topology) through molecular data and graphs

<ins>tanimoto similarity</ins>: quantifiable value that calculates the similarly between two molecular fingerprints

<ins>t-distributed stochastic neighbor embedding (t-SNE)</ins>: a technique that reduces the high-dimensional molecular data into 2D and 3D visualizations

<ins>ROC_AUC (Receiver Operating Characteristic - Area Under Curve)</ins>: a metric used to quantify the accuracy and reliability of a model telling the difference of two different groups

<ins>PRC_AUC (Precision-Recall Curve - Area Under a Curve)</ins>: a metric to quantify the model’s effectiveness in identifying positive cases (in this case, BBB+ molecules)

<ins>MCC (Matthews Correlation Coefficient)</ins>: a quantifiable metric that takes TP (true positives), TN (true negatives), FP (false positives), and FN (false negatives) to score the overall correlation between predicted vs actual results

<ins>BACC (Balanced Accuracy)</ins>: a metric that quantifies the accuracy of the model’s sorting of two different groups (example: positive and negative data groups)

<ins>RF(ECFP) (Random Forest with Extended-Connectivity Fingerprint)</ins>: a type of model that uses molecular structures and bonds to predict a molecule’s properties

<ins>MLP(ECFP) (Multi-Layer Perceptron with ECFP)</ins>: a type of model that uses ECFP (similar to RF[ECFP]) to predict a molecule’s properties

<ins>RF(PCP) (Random Forest with Physiochemical Properties)</ins>: a type of model that takes physiochemical (physical + chemical) properties of a molecule as inputs to predict other properties of a molecule

<ins>MLP(PCP) (Multi-Layer Perceptron with PCP)</ins>: a type of model that uses PCP (similar to RF[PCP]) to predict other properties of a molecule 

<ins>Background</ins>: 

Determinants for BBB permeability include CNS activity, BBB+-, and logBB ratio. Multiple determinants are used as an attempt to enlarge the size and quality of the database. 

    - Problem: Overfitting, too many molecular descriptors for each molecule, can cause the prediction model to learn false and dubious patterns

    - Solution: Instead, deep learning techniques can select optimal and important features and descriptors for BBB permeability 

GNN and Attentive FP were used as models to predict BBB permeability

<ins>Method</ins>: 

MoleculeNet provides information about parameters such as BBB permeability. The BBBp dataset either has a BBB+/BBB- value or a logBB value. M-data (main data) is curated through MoleculeNet and processed.

<ins>Cleaning Data and Defects</ins>: Remove data with the following features

    - SMILES can’t be identified in RDKit 

    - Duplicates 

<ins>Data Curation</ins>: 

    - Remove salts and solvents 

    - Neutralize 

    - If a SMILES contains multiple molecules, separate the compound and extract the molecule with the largest molecular weight (often the “main drug”/”parent drug”)

<ins>Supplementary Data Set</ins>:

S-data (supplementary data) is a combination of external data containing BBB permeability values used to test the accuracy of the existing BBBp model.

    - Validity of Divergence between S-data and M-data: To ensure that the S-data is significantly/comparably different from the M-data, Tanimoto similarity and t-distributed stochastic neighbor embedding (t-SNE) are used.

<ins>Models</ins>:

GROVER and Attentive FP models were chosen to be trained for BBB permeability because they are pre-trained with existing data and use molecular fingerprints and molecular structure to make more accurate decisions about BBB permeability.

<ins>Results</ins>: 

After testing with S-data, GROVER, and Attentive FP, on average, better performance for ROC_AUC, PRC_AUC, MCC, and BACC in comparison to other models: RF(ECFP), MLP(ECFP), RF(PCP), and MLP(PCP). 

<ins>Attempt to Account for Active Transport</ins>:

Because models so far only predicted the passive transport of drugs using molecular data and not the active transport of drugs, information about the substrates of transporters was used as an attempt to train the molecules for active transport as well. However, GROVER had similar results with the other predictive models, showing the lack of capability for GROVER to successfully use the data to account for the active transport of drugs through the BBB 
 
<ins>Limitations</ins>:

The deviation of data (the provided data for models contained three times as many BBB+ molecules in comparison to BBB- molecules, when in reality, 98% of drugs can’t pass through the BBB), algorithms, and data quantity limited the accuracy of the model. 

<ins>Uncertainty Estimations</ins>:

As an attempt to account for the limitations, uncertainty estimations were used to predict the reliability of the prediction

<ins>Algorithms for Uncertainty Estimation</ins>: 

Five proposed algorithms were used for uncertainty estimation of the result:

    - Entropy

    - MC-dropout

    - Multi-initial

    - FPsDist

    - LatentDist

<ins>Results</ins>: An ensemble of Entropy and MC-dropout provided the best results for uncertainty estimation.

<ins>Result with Uncertainty Estimation</ins>: Combined with the algorithms for the uncertainty estimate, as well as the use of the MCC graph to calculate the prediction performance of the models, GROVER’s performance was ultimately enhanced. Finally, the GROVER-BBBp model was chosen as the ideal model and was tested with clinical and marketed drugs.
