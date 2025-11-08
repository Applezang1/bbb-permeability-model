# Chapter 3: Shallow Neural Networks 

## Key Concepts and Words: 

**<ins>Input layer</ins>**: the layer in a multi-layer perceptron that consists of the input data 

**<ins>Hidden layer</ins>**: the layer in a multi-layer perceptron that consists of the hidden units

**<ins>Output layer</ins>**: the layer in a multi-layer perceptron that consists of the output data 

**<ins>Pre-activations</ins>**: The value of the hidden layer before it undergoes an activation function 

**<ins>Activation</ins>**: The value of the hidden layer after it undergoes an activation function 

**<ins>Multi-layer perceptron (MLP)</ins>**: a neural network that consists of at least one hidden layer. Shallow neural networks and deep neural networks are the two classifications of a multi-layer perceptron

**<ins>Shallow neural networks</ins>**: neural networks with only one hidden layer

**<ins>Deep neural networks</ins>**: neural networks with more than one hidden layer 

**<ins>Feed-forward networks</ins>**: a type of neural network where information only flows from the input to the output data 

**<ins>Network weights</ins>**: the magnitude of an effect an input value has on the value of a hidden layer/hidden unit 

**<ins>Bias</ins>**: the parameter that offsets the entire output function in the end

## Shallow Neural Network: 
**<ins>General equation for a shallow neural network</ins>**: y[j] = phi[j0] + summation from d = 1 to D of phi[jd]*h[d] 
<ins>Variables</ins>: 

    - phi[j0] is the parameter that offsets the entire function

    - phi[jd] is the weight of each hidden unit of the function (determines how much of an impact the hidden layer has on the output)

**<ins>Hidden unit equation</ins>**: h = a*[phi[d0] + summation from i = 1 to D[i] of phi[di]*x[i] ]
<ins>Variables</ins>: 

    - a = activation function
    
    - d = number of hidden units
    
    - x, y = multi-dimensional inputs/outputs

## Shallow Neural Network Example: 
A function in terms of x and 10 parameters: y = phi[0] + phi[1]*a*(phi[10] + phi[11x]) + phi[2]*a*(phi[20] + phi[21]x) + phi[3]*a*(phi[30] + phi[31]x)

<ins>Variables</ins>: 

    - phi: the parameters
    
    - a: the activation function

## General Method for Shallow Neural Network:
The linear function of the input data is computed 

    - Note: A linear function of the input data is a function with an input variable (Example: phi[20] + phi[21]x)

The value of the linear function is passed through an activation function (a)

<ins>Example</ins>: 

    - ReLU (rectified linear unit): a function that returns the value when the input is positive and returns a zero in any other scenario

The value of the activation function is offsetted by a parameter (phi[0])

We can optimize the value of the parameters to minimize the least squares loss function between the input/output data pair and the predicted value to train the model

## Hidden Units  
<ins>Function</ins>: Each hidden unit adds a component that contributes to the shape of the final/resulting function 

In the linear function mentioned above, the hidden units h1, h2, and h3 are 

    - h1 = a*(phi[10] + phi[11]x)

    - h2 = a*(phi[20] + phi[21]x)

    - h3 = a*(phi[30] + phi[31]x) 

Simplified Equation: y = phi[0] + phi[1]*h1 + phi[2]*h2 + phi[3]*h3 
### Classification of Hidden Units:

    - Active: A hidden layer contributes to the output 

    - Inactive: A hidden layer doesn’t contribute to the output 

## Universal Approximation Theorem 

**<ins>The Universal Approximation Theorem</ins>** states that a shallow network with enough hidden units can fine-tune the function to a point where it can approximate any complex function with precision 

## Shallow Neural Network for Multivariate Inputs/Outputs: 
<ins>Two Scenarios</ins>: 

    - #1: A function has a multivariate (more than one) output 

    - #2: A function has a multivariate input and a multivariate output

In either of these scenarios, each hidden layer returns a piecewise linear function that contributes to the shape of the continuous piecewise linear function of the input (output function)  


