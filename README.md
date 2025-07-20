# Neural-Network-
Documentation of me building a Neural Network from scratch. Below you can find some useful theory that I researched from multiple sources that add context to some steps in my practice neural network.

# Calculations that were too long to be code comments.
## Backwards Prop Cost Gradient Calculation 
### (Sources: 
### Medium article - https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc, 
### Stack Exchange thread on the derivative of matrix multiplication -https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product)

The only independent variables in a neural network are the weights and biases associated with each layer. As such, the gradient vector of the cost function should only consist of these variables:

<img width="674" height="467" alt="image" src="https://github.com/user-attachments/assets/fc2e5231-6f42-44b0-99c0-25b80d5c672f" />

(Source: Linked Medium Article)

### The Derivative of a Matrix
