# Paper 4: DeePred-BBB: A Blood Brain Barrier Permeability Prediction Model With Improved Accuracy

<ins>Paper 4 Link</ins>: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.858126/full#supplementary-material 

## Key Terminology for Contextual Understanding: 

**<ins>Support Vector Machines (SVM)</ins>**: an algorithm that quantifies the parameters into vectors and graphs it on a hyperplane (a plane that is one dimension less than the space dimension) to find the distance between the vector and the hyperplane, which measures the uncertainty value

**<ins>Artificial Neural Networks (ANN)</ins>**: an algorithm that learns the weights (example: molecular features) essential to BBB permeability through adjustment of its value and outputs its permeability based on these features.

**<ins>K-nearest neighbors (kNNs)</ins>**: a learning algorithm where a molecule’s features are represented as a vector, where the similarity of the molecules can be determined through their k-distance (commonly Euclidean distance). This is then used to predict BBB permeability

**<ins>Naive Bayes (NB)</ins>**: a learning algorithm where the NB algorithm calculates the probability of each feature being BBB permeable or non-permeable and combines these probabilities over all of the molecule’s features to determine BBB permeability

**<ins>Random Forests (RFs)</ins>**: an algorithm where decisions are split up into multiple trees based on different initial parameters or subsets of data, and combined to determine BBB permeability

**<ins>DNN</ins>**: a type of ANN with more layers, which allows it to adjust the weights of more complex data and interactions of molecules in order to find relationships. Then, BBB permeability is determined through the adjustment of the weights

**<ins>CNN-1D</ins>**: a type of DL (deep learning model) that takes inputs of 1-dimensional molecular data and uses kernels (an array of different values for weights) to identify local patterns and changes. After, the most important data is extracted and used to predict BBB permeability.

**<ins>CNN (VGG16)</ins>**: a type of CNN with 16 layers where the first 13 layers use kernels to detect local patterns and changes in molecular data, and then extract the most important features. The last three layers use the combined features to determine BBB permeability.

**<ins>SVM (kernel function)</ins>**: a model that uses kernels to transform data to a higher-dimensional space separated by a hyperplane. The distance between the vector and the hyperplane is the uncertainty of the prediction, which can ultimately be used to determine BBB permeability.

<ins>Background</ins>:

Predicting BBB permeability using computational algorithms is inexpensive compared to clinical experiments and has been attempted multiple times before through the use of artificial intelligence.

Some Models Include: 

    - Support Vector Machines  (SVM)

    - Artificial Neural Networks (ANN)

    - K-nearest neighbors (kNNs) 

    - Naïve Bayes (NB) 

    - Random Forests (RFs)

<ins>Method</ins>: 

Data is curated to remove redundant compounds and label them as non-permeable and permeable. The dataset is split into a training and test set (3:1 ratio).

## Data Values and Features:

<ins>Features of Molecules in datasets included</ins>:

    - Physiochemical properties

    - Molecular Access System Fingerprints (MACCS)

    - Substructure fingerprints were used as the features of each molecule 

Features were used for algorithms: ML, DNN, CNN-1D

<ins>Development of Prediction Models</ins>: 

ML-based algorithms (SVM [kernel], kNN, RF) and DL-based algorithms (DNN, CNN-1D, CNN (VGG16)) were used as potential algorithms for the predictive model. 
Analyzing: DL algorithm was analyzed by the “tenfold cross-validation method” (training data divided into 10 sets, 1 set used as test, and the rest used to train the model) 

    - Convolutional Neural Network - 1 Dimension (CNN-1D) 

    - Deep Neural Network (DNN)

    - Convolutional Neural Network by VGG16 Transfer Learning (CNN-VGG16)   

<ins>Result Interpretation</ins>:

Results were quantified and validated through:

    - Area Under Curve (AUC): a quantitative value of how accurate the model is at distinguishing which molecules are BBB permeable or not

    - Area Under the Precision-Recall Curve (AUCPR): a value that measures the precision and recall of the model for identifying BBB+ molecules

    - Average Precision (AP): a value that measures the model’s precision of BBB permeability as it encounters more data (or the recall increases)

    - F1 Score: a value that shows the precision of the result and the recall of the BBB+ molecules

    - Hamming distance (HD): the number of incorrect predictions of BBB permeability 

Note: 

    - Recall: (TP)/(TP+FN)

    - Precision: (TP)/(TP+FP)

Where TP: true positives, FN: false negatives, FP: false positives 

<ins>Results</ins>: SVM was the best ML-based algorithm. DNN was the best DL-based algorithm that was superior to SVM. 

<ins>Possible Reasoning</ins>: DNN can handle large datasets more effectively and accurately than SVM

<ins>Conclusion</ins>: DNN-based “DeePred-BBB” was used as the model to predict BBB permeability. In order for the DNN model to be used, the PaDel tool was first used to calculate the molecular features. The molecular features were fed to the DNN model as input, which it then used to make a prediction about BBB permeability.
