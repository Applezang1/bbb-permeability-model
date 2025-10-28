# Chapter 1: Introduction   

## Artificial Intelligence
**<ins>Artificial Intelligence</ins?**: systems that are designed to simulate intelligent behavior 

**<ins>Machine learning</ins>**: subclass of AI, a type of AI model that takes inputted data and fits the data into a mathematical model in order to make decisions and predict an output  

        - Machine learning methods:

            1. Sets of input/output pairs are given 

            2. A family of equations are chosen to map the input to the output based on which equation most accurately fits the set of input/output pairs (called training, fitting a model) 

            3. An output is given

**<ins>Deep neural network (deep network)</ins>**: subclass of machine learning, a type of machine learning model that predicts outputs through multiple layers that process inputs 

    - Function: Deep neural networks often define a set of equations that can formulate relationships between inputs and outputs whose relationship are very broad (mixed-type relationships) 

<ins>Multivariate Binary Example</ins>: picture with a cow -> pixels that only contains cows

<ins> Multivariate Regression Example</ins>: image of a street scene -> depth at each pixel

**<ins>Deep learning</ins>**: process of training and fitting a model onto the data 

**<ins>Input Data Characteristics</ins>**: 
    - <ins>Data Type</ins>: discrete/continuous
    - <ins>Dimensionality</ins>: low dimensional/high dimensional
    - <ins>Length</ins>: constant or variable length  
    - <ins>Note</ins>: Latent variables express input data in a simplified way by observing correlations or patterns between the data. Latent variables are useful because of their benefits, which require less input data and cause higher accuracy in predicted output

## Types of Machine Learning Methods: 
<ins>Types of Machine Learning Methods</ins>: supervised, unsupervised, reinforcement learning 

### Supervised Learning
**<ins>Supervised learning</ins>**: a type of machine learning model that learns to map inputs to outputs 

    - Note: The outputs and inputs are usually outputted/inputted as a vector of numbers 

**<ins>Types of Supervised Learning Models</ins>**: 

<ins>Regression</ins>: supervised learning model that returns continuous numbers 

    - Example: Characteristics of a house -> Predicted price

<ins>Multivariate regression</ins>: supervised learning model that returns more than one number 

    -Example: Molecule structure -> Freezing and Boiling Point

<ins>Binary classification</ins>: supervised learning model that returns a binary number (0 or 1) 

    - Example: Question -> Yes or No

<ins>Multiclass classification</ins>: supervised learning model that returns a classification from a list of 2+ classification categories 

    - Example: Audio file -> Genre of music 

### Unsupervised Model
**<ins>Unsupervised model</ins>**: a type of machine learning model where the given data doesn’t have an output label. An unsupervised model aims to describe or understand the structure of the given input data 

    - Example: Completing the next word in a sentence, generating a continuation of a story

### Reinforcement Learning
**<ins>Reinforcement Learning</ins>**: a machine learning model that chooses from a set of possible actions each with rewards or punishments. The overall goal of a model with reinforcement learning is to learn to maximize its rewards. This is done through a policy network.

All machine learning models with reinforcement learning must formulate which past action caused the most future rewards. This is known as the temporal credit assignment problem

<ins>Temporal credit assignment problem</ins>: a challenge in reinforcement learning models where the reward is given some time after the action is taken. Therefore, it is unclear which past action resulted in the gain in reward and the models must learn to correlate which past action resulted in the reward.

<ins>Key Problem in Reinforcement Learning</ins>: Exploration (process of trying different actions for possible better rewards) vs  Exploitation (abusing a strategy that it already knows for consistent rewards) 

<ins>Example of Exploration vs Exploitation</ins>: 

    - Background: A robot must move through an obstacle course by performing actions. A reward is given if the robot reaches the end of the obstacle course.  

        - Scenario: The robot learns how to navigate to the end by laying down and pushing with one leg 

            - Exploration: Should the robot discover other methods of movement for a possible increase in rewards?

            - Exploitation: Should the robot keep exploiting the current method for consistent rewards?