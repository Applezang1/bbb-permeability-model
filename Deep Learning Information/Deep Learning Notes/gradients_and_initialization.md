# Chapter 7: Gradients and Initialization 

## Key Words and Concepts: 

**<ins>Forward Pass</ins>**: the act of running the neural network model and storing all the activation value (output of the hidden units) and the pre activations for each layer for each input data 

**<ins>Backward Pass</ins>**: the act of computing how a change to a weight or bias affects the output of a neural network mode by computing the change of the hidden layer that the weight/bias feeds into and how a change in a hidden unit in that hidden layer affects the values of previous hidden layers.

## Backpropagation Algorithm
In stochastic gradient descent/gradient descent, we need to be able to effectively calculate the partial derivative of the loss function in terms of each parameter (bias and weight parameters)  

<ins>Two Requirements</ins>: Before computing the backpropagation algorithm, the following two requirements needs to be carried out 

    - To determine how each parameter (bias, weight) influences the output of the hidden unit and the output of the neural network, we need to compute the forward pass 

    - To determine how changes to a weight or bias affects the output of the neural network model, we need to compute the backward pass

<ins>Example</ins>: 

Compute the forward pass and store the equations and values of the pre activation and the activation/hidden unit output 

Compute the backward pass by: 

    - Finding the derivative of the loss function in respect to the output value equation. 

    - Go backwards by finding the derivative of the loss function in respect to the activation function/hidden unit output, which was used in the output value equation. 

        - This can be done using the chain rule and multiplying the derivative of the output value equation in respect to the activation/hidden unit output and multiplying that by the derivative of the loss function in respect to the output value equation.  

    - Repeat the backward pass till you have found the derivative of the loss function in respect to all the pre activation and activation equations 

Using the derivative of the loss function in respect to the pre-activation equations, compute the backward pass again by: 

    - Finding the derivative of the loss function in respect to each of the parameters in the pre-activation equations. 

        - This can be computed using the chain rule and multiplying the derivative of the loss function in respect to the pre activation and multiplying that by the derivative of the pre activation in respect to the parameter 

    - Repeat the backward pass till you have found the derivative of the loss function in respect to all the parameters 

<ins>Note</ins>: This example can be to many-layered hidden networks, with the only difference being that the intermediate variables (pre-activation, activation) as well as the parameters (weights, biases) are vectors instead of a single variable 

<ins>Note</ins>: If the training data is 1D (vector), the input for backpropagation is a 2D tensor (matrix) which consists of the training data and the data dimension (this is done for faster and better computation) 

## Parameter Initialization 
Not choosing sensible parameters can cause the pre-activations to get infinitely small or infinitely big, causing either the vanishing gradient problem (computed gradients are too small) or the exploding gradient problem (computed gradients are too big) which are not ideal for training 

Therefore, we need to ensure that the variance between the chosen parameters isn’t too big or too small in order to avoid the gradient problems, use the equation: 

    - variance^2 = 2/Dh (where Dh= dimensions of original layer)

In order to compute the ideal variance between the chosen parameters 

If there are a different number of hidden units in each hidden layer, use the equation: 

    - variance^2 = 4/(Dh + Dh’) (where Dh and Dh’ is the dimensions of the two layers)
