# Comparison between Learning Models
## Types of Machine Learning Methods: Supervised, Unsupervised, Reinforcement learning 
### Supervised learning: 
**<ins>Supervised learning</ins>**: Supervised learning models are a type of learning model that maps specific input values to output values. 

<ins>Learning Mechanism</ins>: 

    - A supervised learning model learns how to map specific input values to output values by using a training dataset with known relationships between input and output

    - The model makes its own predictions with the given input values and computes the loss function between its predicted output value and the actual output value

    - The model then updates its parameters to minimize the loss function, allowing the model to better fit the known relationship

<ins>Example</ins>: A model that predicts house prices given properties about the house 

<ins>Strengths</ins>: Because supervised learning models are given a training dataset with known relationships between input and output values, they allow the model to achieve high accuracy. 

<ins>Weakness</ins>: Supervised learning models require large training datasets to perform well, which are difficult to obtain

### Unsupervised Learning
**<ins>Unsupervised model</ins>**: Unsupervised learning models are a type of learning model that learns to find specific patterns in input data

<ins>Learning Mechanism</ins>:  

    - Unsupervised learning models learn how to find patterns in input data through loss functions, where each unsupervised learning model is unique due to their internal loss function.

    - The within-cluster sum of squares is an example of a loss function that certain types of unsupervised learning models use

    - This loss function measures how close each data point is to the cluster. 

    - The parameters are updated to move the cluster closer to its neighboring data points

<ins>Example</ins>: Grouping pixels that illustrate the same object  

<ins>Strengths</ins>: Unsupervised learning models don’t require training datasets to learn and interpret input data 

<ins>Weaknesses</ins>: The lack of a training dataset makes it difficult to validate whether the loss function correctly optimizes the parameters, and external validation methods need to be used to check the accuracy of the output

### Reinforcement Learning
**<ins>Reinforcement Learning</ins>**: Reinforcement learning models are a type of learning model that learns decision-making strategies in a given environment  

<ins>Learning Mechanism</ins>: 

    - Reinforcement learning models are given a set of actions, which are randomly chosen at first 

    - The model either receives a reward or a punishment based on its actions 

    - Based on the output from its actions, the model updates its policy (learned action plan) to optimize its rewards

<ins>Example</ins>: A model that learns how to play chess 

<ins>Strengths</ins>: The model doesn’t require any training dataset and can use its own previous actions to train and optimize its parameters 

<ins>Weaknesses</ins>: The model can run into the exploration-exploitation trade-off, where a sufficient reward keeps the model from experimenting to possibly increase its reward.

