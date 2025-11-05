# Chapter 2: Supervised Learning 

## Key Concepts and Words:

**<ins>Tabular/Structured Data</ins>**: A form of data that is structured like tables with rows and columns

**<ins>Inference</ins>**: The process of using an input data to compute a prediction/output 

**<ins>Parameter</ins>**: a variable that affects the relationship between an input and an output 

**<ins>Loss function</ins>**: a function that calculates the deviation from the predicted output value to the actual output value for one set of input and output value

**<ins>Cost function</ins>**: a function that calculates the deviation from the predicted output value to the actual output value for the entire dataset

**<ins>Generative model</ins>**: a machine learning model that takes in an input from real-world measurements and returns an output 

**<ins>Discriminative model</ins>**: a machine learning model that uses the value of an output y and returns input data from real-world measurements

## Supervised Learning Model
**<ins>Supervised learning model</ins>**: a type of machine learning model that maps one or more inputs to one or more outputs 

    - Example: Age and mileage of a car -> the car’s estimated price 

<ins>Overview of Method</ins>: Supervised learning models usually use a mathematical function to compute the output based on the input data (inference) while changing the parameters of the mathematical function to adjust the output (training) 

## Overview of Developing a Supervised Learning Model: 

    - Input a training dataset of input and output pairs 
    
    - Define a scalar value (loss L) that quantifies the difference between the predicted output value from the mathematical model and the actual output value

    - Minimize the loss functions by adjusting the parameters of the mathematical model. The loss function is a function in terms of a set of parameters

## Example of Supervised Learning Model: 
**<ins>1D Linear Regression Model</ins>**: a straight line equation that has one input (x) and one output (y). The 1D Linear Regression Model has two parameters: y-intercept and slope.

### Computing the Loss Function:

    - Define a training error which is the deviation between the model’s prediction of the output value and the actual output value (this type of training error is called a least-squares loss) 

    - This creates the loss function, which is a function dependent on the parameters, with the most optimal parameters minimizing a loss function 

### Training the Model: 

**<ins>Gradient</ins>**: geometric visualization of the rate of change of a muti-variable function. The gradient has different heights at different locations and the point of the lowest height minimizes the multi-variable function

By calculating the gradient of the loss function, we can move down in the direction that’s most steeply downhill until the gradient is flat and we have reached a minimum 

### Testing the Model: 

A separate set of data (test data) can be used to calculate/determine the accuracy of the supervised learning model.

### Two Possible Results from Testing: 

    - Underfitting: the model is unable to perform well for both the test data and the training data. This concludes that the model is unable to capture the relationship between input and output values

    - Overfitting: the model performs well with the training data but not with the test data. This concludes that the model is overly complex and doesn’t generalize well (small changes to data causes a huge loss in accuracy)