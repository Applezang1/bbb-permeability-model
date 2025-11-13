# Chapter 4: Deep Neural Networks  

## Key Concepts and Words: 

**<ins>Width</ins>**: the number of hidden units in each layer of a neural network

**<ins>Depth</ins>**: the number of hidden layers in a neural network

**<ins>Capacity</ins>**: the total number of hidden units in a neural network 

**<ins>Hyperparameter</ins>**: quantities that defines the model and are chosen before training a model (example: number of layers, number of hidden units)

## Composition of Shallow Neural Network 
Shallow neural networks can be composed so that the output of the first shallow neural network acts as the input of the second shallow neural network 

### Composed Shallow Neural Network vs Normal Shallow Neural Network
<ins>Scenario</ins>: Shallow neural network with 6 hidden units (6 shallow neural network) vs shallow neural network with 3 hidden units whose output serves as the input for the second shallow neural network with 3 hidden units (3+3 neural network) 

<ins>Result</ins>: The 3+3 neural network is able to model more complex relationships than the 6 shallow neural network since each hidden unit in the 3+3 neural network is split off to form 3 more hidden units, therefore there will be more linear regions in a given x interval for the 3+3 neural network to use to predict the output compared to the 6 shallow neural network. 

## Deep Neural Network 
<ins>Definition</ins>: A deep neural network is a type of neural network with more than one hidden layer

### Deep Neural Network (2 hidden layers) 
    - The output of the first network (y = phi0 + phi1*h1 + phi2*h2 + phi3*h3) is used as the input of the second layer.  
    - Each constant or hidden unit (h1, h2, h3) is multiplied by a new set of arbitrary constants and is inputted into an activation function 
    - The output of the activation function is the hidden units of the second layer, whose value depends on the hidden units of the first layer 

## Shallow vs Deep Neural Networks 

1) The Universal Approximation Theorem applies for both neural networks, therefore a shallow or deep neural network can model any complex function with enough hidden units 

2) Deep neural networks can create much more linear regions than shallow neural networks given the same dimensions of input, the same dimensions of output, the number of parameters, and the number of hidden units 

3) Deep neural networks have much more depth efficiency (better approximations for certain functions) than shallow neural networks  

4) Deep neural networks better capture local-to-global processing 

5) Deep neural networks are easier to train and generalizes (predicts) data better than shallow neural networks
