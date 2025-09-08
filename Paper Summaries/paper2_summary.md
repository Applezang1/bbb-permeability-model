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
