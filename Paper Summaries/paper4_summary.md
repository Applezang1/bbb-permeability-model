# Paper 4: DeePred-BBB: A Blood Brain Barrier Permeability Prediction Model With Improved Accuracy

<ins>Paper 4 Link</ins>: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.858126/full#supplementary-material 

## Key Terminology for Contextual Understanding: 

<ins>Support Vector Machines (SVM)</ins>: an algorithm that quantifies the parameters into vectors and graphs it on a hyperplane (a plane that is one dimension less than the space dimension) to find the distance between the vector and the hyperplane, which measures the uncertainty value

<ins>Artificial Neural Networks (ANN)</ins>: an algorithm that learns the weights (example: molecular features) essential to BBB permeability through adjustment of its value and outputs its permeability based on these features.

<ins>K-nearest neighbors (kNNs)</ins>: a learning algorithm where a molecule’s features are represented as a vector, where the similarity of the molecules can be determined through their k-distance (commonly Euclidean distance). This is then used to predict BBB permeability

<ins>Naive Bayes (NB)</ins>: a learning algorithm where the NB algorithm calculates the probability of each feature being BBB permeable or non-permeable and combines these probabilities over all of the molecule’s features to determine BBB permeability

<ins>Random Forests (RFs)</ins>: an algorithm where decisions are split up into multiple trees based on different initial parameters or subsets of data, and combined to determine BBB permeability

<ins>DNN</ins>: a type of ANN with more layers, which allows it to adjust the weights of more complex data and interactions of molecules in order to find relationships. Then, BBB permeability is determined through the adjustment of the weights

<ins>CNN-1D</ins>: a type of DL (deep learning model) that takes inputs of 1-dimensional molecular data and uses kernels (an array of different values for weights) to identify local patterns and changes. After, the most important data is extracted and used to predict BBB permeability.

<ins>CNN (VGG16)</ins>: a type of CNN with 16 layers where the first 13 layers use kernels to detect local patterns and changes in molecular data, and then extract the most important features. The last three layers use the combined features to determine BBB permeability.

<ins>SVM (kernel function)</ins>: a model that uses kernels to transform data to a higher-dimensional space separated by a hyperplane. The distance between the vector and the hyperplane is the uncertainty of the prediction, which can ultimately be used to determine BBB permeability.