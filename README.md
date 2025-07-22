# Neural-Network-
Documentation of me building a Neural Network from scratch. Below you can find some useful theory that I researched from multiple sources that add context to some steps in my practice neural network.

# Calculations that were too long to be code comments.
### (Sources: 
### Medium article - https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc, 
### Stack Exchange thread on the derivative of matrix multiplication -https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
### Stack Exchange thread on derivatives based on matrices - https://math.stackexchange.com/questions/622195/taking-a-derivative-with-respect-to-a-matrix
### Matrix Calculus Article - https://en.wikipedia.org/wiki/Matrix_calculus
### Element wise matrix division - https://math.stackexchange.com/questions/172248/notation-for-element-wise-division-of-vectors)

## Backwards Prop Cost Gradient Calculation 
The only independent variables in a neural network are the weights and biases associated with each layer. As such, the gradient vector of the cost function should only consist of these variables:

<img width="674" height="467" alt="image" src="https://github.com/user-attachments/assets/fc2e5231-6f42-44b0-99c0-25b80d5c672f" />

### The Derivative of a Scalar With Respect To a Matrix

The derivative of a scalar function with respect to a matrix creates a matrix of the same dimensions, where each i-jth element of that new matrix is the partial derivative of the function with respect to the i-jth entry of the original matrix. For some scalar function f(A) where A is a matrix:

<img width="919" height="408" alt="image" src="https://github.com/user-attachments/assets/4ba0a21c-faf0-496e-91f0-780f051f93e5" />


For upcoming calculations, it is better to understand this transformation through the total differential of the scalar function. If f(D) is a scalar function where D is a matrix, then:

<img width="1226" height="366" alt="image" src="https://github.com/user-attachments/assets/c8e1e259-ef44-44f6-9e37-0b7a831439ab" />

The total differential f in this case is essentially the total change in the function output when an infinitesimal change occurs in each entry of the origninal matrix D.

### Calculating Gradient Vector Components
In this section, I will derive the the partial derivative of cost with respect to A (the activated output data), Z (the raw output data), W (the weight matrix), and b (the bias matrix) for arbritrary layers in the neural network.

The neural network that I use for practice and the one that is used in the linked medium article has the following chain rule tree for its cost function:

<img width="792" height="742" alt="image" src="https://github.com/user-attachments/assets/8a4b273f-f6d9-4c0d-8663-c758e0535471" />

Note that C is the cost function, A3 is equivavlent to y_hat, and A0 is the raw input training data.

#### ∂C/∂A 

The first partial derivative that must be computed is ∂C/∂A at the output layer. Let's call this layer capital L and the output A^[L]. This should result in an n^[l] x m matrix (where n^[l] is the number of nodes in layer l and m is the number of training samples) where each i-jth entry is  ∂C/∂aᵢⱼ. Therefore, one can find this partial derivative matrix by finding the arbritrary ∂C/∂aᵢⱼ. This neural network only has one output node so there is only 1 row in the output matrix. Therefore, the derivative can be shortened to ∂C/∂aⱼ The cost function is:

<img width="1339" height="212" alt="image" src="https://github.com/user-attachments/assets/d12aa61b-9094-4984-a2ec-87583bf90c9a" />

taking the derivative ∂C/∂aⱼ in the L layer: 

<img width="1208" height="366" alt="image" src="https://github.com/user-attachments/assets/9bf980df-4ea4-46e0-9aab-9fdc9d0902df" />

Notice that all terms where k ≠ j cancel out when taking the partial derivative with respect to aⱼ. Therefore:

<img width="1446" height="446" alt="image" src="https://github.com/user-attachments/assets/a991a315-8cd6-41b0-b812-1beb95e752c5" />

Or in matrix form:

where ⊘ is the element wise division (Hadamard division):

<img width="1099" height="210" alt="image" src="https://github.com/user-attachments/assets/dc58eebe-f103-4826-bdfb-62d1d3230788" />

