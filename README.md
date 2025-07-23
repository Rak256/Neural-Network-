# Neural-Network-
Documentation of me building a Neural Network from scratch.

This neural network is a binary classification network. I cover all steps of creating this network (Forward Propagation, Cost function, Back Propagation) in my code through comments or through this README file for calculations that are larger and more complex.

Below you can find some useful theory that I researched from multiple sources that add context to some steps in my practice neural network.

# Calculations that were too long to be code comments.
### (Sources: 
### Medium article - https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc, 
### Stack Exchange thread on the derivative of matrix multiplication -https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
### Stack Exchange thread on derivatives based on matrices - https://math.stackexchange.com/questions/622195/taking-a-derivative-with-respect-to-a-matrix
### Matrix Calculus Article - https://en.wikipedia.org/wiki/Matrix_calculus
### Element wise matrix division - https://math.stackexchange.com/questions/172248/notation-for-element-wise-division-of-vectors)

NOTE: Matrix multiplication between some matrices A and B is represented in this documentation as AB.

## Backwards Prop Cost Gradient Calculation 
The only independent variables in a neural network are the weights and biases associated with each layer. As such, the gradient vector of the cost function should only consist of these variables:

<img width="674" height="467" alt="image" src="https://github.com/user-attachments/assets/fc2e5231-6f42-44b0-99c0-25b80d5c672f" />

### The Derivative of a Scalar With Respect To a Matrix

The derivative of a scalar function with respect to a matrix creates a matrix of the same dimensions, where each i-jth element of that new matrix is the partial derivative of the function with respect to the i-jth entry of the original matrix. For some scalar function f(A) where A is a matrix:

<img width="919" height="408" alt="image" src="https://github.com/user-attachments/assets/4ba0a21c-faf0-496e-91f0-780f051f93e5" />


For upcoming calculations, it is better to understand this transformation through the total differential of the scalar function. If f(D) is a scalar function where D is a matrix, then:

<img width="1226" height="366" alt="image" src="https://github.com/user-attachments/assets/c8e1e259-ef44-44f6-9e37-0b7a831439ab" />

The total differential ∂f in this case is essentially the total change in the function output when an infinitesimal change occurs in each entry of the origninal matrix D. This is why we sum over the rows and columns of D.

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

where ⊘ is the element wise division (Hadamard division) operator:

<img width="1099" height="210" alt="image" src="https://github.com/user-attachments/assets/dc58eebe-f103-4826-bdfb-62d1d3230788" />

#### ∂A/∂Z

Since A is achieved through element-wise operations on Z, we can get ∂A/∂Z by taking the partial derivative of aᵢⱼ with respect to zᵢⱼ where i and j are arbritrary indexes. Note that the intuition behind this is that only the i-jth element of Z affects the i-jth element of A. The following are the calculations for the top layer, L: 

<img width="955" height="294" alt="image" src="https://github.com/user-attachments/assets/8998c828-22a2-4c5c-b3b0-7b7ab49fa1f0" />
<img width="809" height="756" alt="image" src="https://github.com/user-attachments/assets/b19d3d04-3410-40f1-bee5-56f0d9940f5d" />

Note the L superscript was omitted in the second last line of the equation to not be confused with the ^2 term.
This can also be written in matrix form: 

<img width="749" height="210" alt="image" src="https://github.com/user-attachments/assets/7f44ff62-14b0-443e-a26a-5b44e47727fc" />

Note that the ○ symbol denotes element-wise multiplication (Hadamard multiplication).
#### ∂C/∂Z
Through chain rule, we get:
<img width="1186" height="313" alt="image" src="https://github.com/user-attachments/assets/60b14804-8bc4-4497-9711-770a5f03cbd7" />
<img width="1015" height="713" alt="image" src="https://github.com/user-attachments/assets/c2c77e74-9d4d-4497-a90a-c684bdd95a7b" />

#### ∂Z/∂W
We know that for some layer l:

<img width="848" height="402" alt="image" src="https://github.com/user-attachments/assets/4029358b-020e-4393-9f4c-da0e3d1735a5" />


where i,j,p, and q are arbritrary indexes. Asssume that the column size and and row size of matrices W and A are r respectively.

The biggest question I had during my research was "How do I take the derivative of matrix multiplication?". This is where the total differential of the cost function is very useful.
 
<img width="1179" height="705" alt="image" src="https://github.com/user-attachments/assets/e0d38938-d8d2-46f6-b531-1ea8263c40f5" />

Changing any entry in the weight matrix does affect the bias matrix. Therefore, it's derivative with respect to some pq entry in the weight matrix is always 0. Furthermore, notice that whenever i ≠ p the partial derivative is zero. Therefore, the summation over the rows of Z collapses. Lastly, when k ≠ q, ∂(Wᵢₖ Aₖⱼ)/∂Wₚq is 0. Therefore we get 

<img width="1187" height="592" alt="image" src="https://github.com/user-attachments/assets/e445f586-48cc-4a7c-94f5-9f652b5d6eac" />

