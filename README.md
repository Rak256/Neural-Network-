# Neural-Network-
Documentation of me building a Neural Network from scratch.

This neural network is a binary classification network that takes weight and height data to predict whether or not someone is at a risk of cardiovascular disease. I cover all steps of creating this network (Forward Propagation, Cost function, Back Propagation) in my code through comments or through this README file for calculations that are larger and more complex.

Below you can find some useful theory that I researched from multiple sources that add context to some steps in my practice neural network.

# Prerequisites
- Linear Algebra 1
- Calculus 2
- Neural Networks - Forward propagation (you can look through the code I write to understand this)
- (Optional) Probability and Statistics

NOTE: Matrix multiplication between some matrices A and B is represented in this documentation as AB.

# Brief Intro
This section gives a brief intro into some math and neural network concepts that may give context to some steps in the Backward Propagation section.

A note on the notation that will be used from this point onwards - Superscripts indicate the layer that a component is in. For example, Wˡ is the weight matrix at layer l. Subscripts indicate a particular element of a matrix. For example, Wᵢⱼ represnts the i-jth element (ith row and jth column element) of the weight matrix W.


## Hadamard Multiplication and Division Operations
For some m x n matrices A and B with arbritrary elements aᵢⱼ and bᵢⱼ respectively, the hadamard multiplication between A and B is:

<img width="1252" height="443" alt="image" src="https://github.com/user-attachments/assets/238086b3-d007-41ab-a22b-6ab291c98b74" />

and hadamard division between A and B is:

<img width="1283" height="634" alt="image" src="https://github.com/user-attachments/assets/87f3e229-5764-4a7e-8863-658ebbf14f58" />

## Element Form of Matrix Multiplication
For an m x p matrix A and a p x n matrix B, the m x n matrix Z = AB can be expressed in element form using arbritrary indices i and j, i < m, and j < n:

<img width="875" height="354" alt="image" src="https://github.com/user-attachments/assets/842153f4-762c-4770-94e6-bf2e1693199b" />

While this may look complicated, it simply represents the dot product between the ith row of A and the jth column of B. Hence, we scan across the columns of A and the rows of B to add up all the terms in the dot product, giving us Zᵢⱼ.

## Brief Intro into Forward Propagation
Since this README file focuses on backward propagation, I highly recommend checking out Aadil's Medium Article on neural networks to visually understand the process. However, to give more context to the backward propagation section, I will provide a brief intro of some concepts here.

**Training Data** - Also known as features, they are the raw input used by the AI to learn and solve a problem at hand.

**Training Labels** - Often shortened to labels, they are the true output to the raw training data.

**Nodes** - Some function that processes data based on weights and biases and produces an output.

**Layers** - A section of the neural netowork that contains a fixed amount of nodes. Data is processed and outputted in each layer and is passed between layers.

**Hidden Layers** - Layers within the black box of a neural network. These layers don't hold much intuitive meaning but they transform the raw data provided at the input layer based on patterns within that data and allow the AI to interpret the input at the output layer.

**Forward Propagation** - In forward propagation, data is passed through the input layer and into the hidden layers within the network where that data is transformed. Once transformed, that data is "activated" to produce a more interpretable output. In the neural network used in my code and this README file, I use the sigmoid function to turn the output into a probability between 0 and 1. Finally the data is passed into the output layer where it is interpreted. The output of each layer is denoted as:

<img width="868" height="197" alt="image" src="https://github.com/user-attachments/assets/0ed743e9-beb3-4f18-a71f-15bca052c35b" />

Where W is the weight matrix, A is the activated output of the previous layer and b is the bias matrix.  

**Cost Function** - The cost of a neural network is a measure of the difference between the AI network's output and the true label of the training data.

### (Optional) - Sidenote on this code's Neural Network Cost function
The cost function for this particular neural network is binary cross entropy loss, also known as log loss. It's idea is based on the concept of Bernoulli trials. Essentially, if your network predicts only 2 outcomes that are independent events, for example spam mail or not spam mail, then the probability that your mails are spam can be modelled through Bernoulli trials.

The probability L of some desired output in the neural network, where y is that desired output and y_hat is the network's output, is: 

<img width="765" height="180" alt="image" src="https://github.com/user-attachments/assets/26b59ae8-4bdd-4c56-9aa5-b6ec671ec095" />
 
Collecting the cost of these trials would result in the multiplication of m samples. To avoid this, we take the logarithm of the this probability L. Now, collecting this cost would result in the summation of m samples instead. We then average this total cost over all samples so that the magnitude doesn't scale with label size.

# Backwards Propagation Cost Gradient Calculation 
### (Sources: 
### Medium article - https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc, 
### Stack Exchange thread on the derivative of matrix multiplication -https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
### Stack Exchange thread on derivatives based on matrices - https://math.stackexchange.com/questions/622195/taking-a-derivative-with-respect-to-a-matrix
### Matrix Calculus Article - https://en.wikipedia.org/wiki/Matrix_calculus
### Element wise matrix division - https://math.stackexchange.com/questions/172248/notation-for-element-wise-division-of-vectors)

The only independent variables in a neural network are the weights and biases associated with each layer. As such, the gradient vector of the cost function should only consist of these variables:

<img width="674" height="467" alt="image" src="https://github.com/user-attachments/assets/fc2e5231-6f42-44b0-99c0-25b80d5c672f" />

*Notation*
- **L**: The hidden layer closest to the output layer. L = 3 in this network.
- **W^[L - k]** : For some whole number k, W^[L- k] is the weight matrix at layer W^[L- k].
- **b^[L - k]** : For some whole number k, b^[L- k] is the weight matrix at layer b^[L- k].

## The Derivative of a Scalar With Respect To a Matrix

The derivative of a scalar function with respect to a matrix creates a matrix of the same dimensions, where each i-jth element of that new matrix is the partial derivative of the function with respect to the i-jth element of the original matrix. For some scalar function f(A) where A is a matrix with arbritrary elements aᵢⱼ, this is illustrated below:

<img width="919" height="408" alt="image" src="https://github.com/user-attachments/assets/4ba0a21c-faf0-496e-91f0-780f051f93e5" />

For upcoming calculations, it is better to understand this transformation through the total differential of the scalar function. If f(D) is a scalar function where D is an m x n matrix with arbritrary elements Dᵢⱼ, then:

<img width="1226" height="366" alt="image" src="https://github.com/user-attachments/assets/c8e1e259-ef44-44f6-9e37-0b7a831439ab" />

The total differential ∂f in this case is essentially the total change in the function output when an infinitesimal change occurs in each element of the origninal matrix D. This is why we sum over the rows and columns of D, to sum up all the infinitesimal change in each Dᵢⱼ element.

## Calculating Gradient Vector Components
In this section, I will derive the the partial derivative of cost with respect to A (the activated output data), Z (the raw output data at some layer l), W (the weight matrix at some layer l), and b (the bias matrix at some layer l).

The neural network that I use for practice and the one that is used in the linked medium article has the following chain rule tree for its cost function:

<img width="567" height="772" alt="image" src="https://github.com/user-attachments/assets/fe6092c4-0bfa-47a1-a76b-75f38742b2ab" />


*Notation*
- **C** : Cost function
- **L** : The hidden layer closest to the output layer. L = 3 in this network.
- **Aˡ** : For some layer l, this is the activated output data of that layer.
- **Zˡ** : For some layer l, this is the raw output data of that layer.
- **Wˡ** : For some layer l, this is the weight matrix of that layer.
- **bˡ** : For some layer l, this is the bias matrix of that layer.

Note that A3 is equivavlent to y_hat, and A⁰ is the raw input training data.

## ∂C/∂A 

The first partial derivative that must be computed is ∂C/∂A at the output layer. Let's call the hidden layer closest to the output layer capital L and the output A^[L]. This should result in an n^[L] x m matrix (where n^[L] is the number of nodes in layer L and m is the number of training samples) where each i-jth element is  ∂C/∂Aᵢⱼ. Therefore, one can find this partial derivative matrix by finding the arbritrary ∂C/∂Aᵢⱼ. This neural network only has one output node so there is only 1 row in the output matrix. Therefore, the derivative can be shortened to ∂C/∂aⱼ The cost function is:

<img width="1339" height="212" alt="image" src="https://github.com/user-attachments/assets/d12aa61b-9094-4984-a2ec-87583bf90c9a" />

Where yₖ is the kth element of the 1 x m labeled output matrix, Y. Notice that Y is the same shape as  A^[L]. This is what allows us to write the partial derivative ∂C/∂A in matrix form since this shape condition is necessary for Hadamard multiplication and division.

taking the derivative ∂C/∂aⱼ in the L layer: 

<img width="1208" height="366" alt="image" src="https://github.com/user-attachments/assets/9bf980df-4ea4-46e0-9aab-9fdc9d0902df" />

Notice that all terms where k ≠ j cancel out when taking the partial derivative with respect to aⱼ. Therefore:

<img width="1446" height="446" alt="image" src="https://github.com/user-attachments/assets/a991a315-8cd6-41b0-b812-1beb95e752c5" />

Or in matrix form:

where ⊘ is the element wise division (Hadamard division) operator:

<img width="1189" height="181" alt="image" src="https://github.com/user-attachments/assets/2ea0fba3-aebe-4086-9d26-5f7473e1b114" />

## ∂A/∂Z

Since A is achieved through element-wise operations on Z, we can get ∂A/∂Z by taking the partial derivative of aᵢⱼ with respect to Zᵢⱼ where i and j are arbritrary indices. Note that the intuition behind this is that only the i-jth element of Z affects the i-jth element of A. The following are the calculations for the hidden layer closest to the output layer, called L: 

<img width="955" height="294" alt="image" src="https://github.com/user-attachments/assets/8998c828-22a2-4c5c-b3b0-7b7ab49fa1f0" />
<img width="799" height="767" alt="image" src="https://github.com/user-attachments/assets/e563f711-b62f-4060-bc1f-11e9ec6b4569" />

This can also be written in matrix form: 

<img width="749" height="210" alt="image" src="https://github.com/user-attachments/assets/7f44ff62-14b0-443e-a26a-5b44e47727fc" />

Note that the ○ symbol denotes element-wise multiplication (Hadamard multiplication).

## ∂C/∂Z
Since A is achieved through element wise operations on Z, we perform Hadamard multiplication in the chain rule.

Through chain rule, we get:
<img width="1252" height="408" alt="image" src="https://github.com/user-attachments/assets/57bd6ced-3102-414e-a777-7c73cf92f2d2" />
<img width="1015" height="713" alt="image" src="https://github.com/user-attachments/assets/c2c77e74-9d4d-4497-a90a-c684bdd95a7b" />

## ∂C/∂W
We know that for some layer l:

<img width="979" height="450" alt="image" src="https://github.com/user-attachments/assets/cdb0e867-3fd9-46c1-866f-83c01ba33531" />

where i and j are arbritrary indices. Asssume that the column size and and row size of matrices W and A are r respectively. You might be wondering why we use bᵢ instead of bᵢⱼ for the bias matrix element in the i-j element form of the above equation. This is because while b is an n^[l] x m matrix, it only consists of the first column vector of the bias matrix (column with j = 1) broadcasted (copied over) to create m columns. This makes sense since each layer should only have 1 bias vector with n^[l] components (1 for each node). This also means that any changes made in the first column will be duplicated in the columns j > 1. So, Zᵢⱼ only depends on bᵢ of the first and original column of the bias matrix.  

The biggest question I had during my research was "How do I take the derivative of matrix multiplication?". This is where the total differential of the cost function is very useful.
 
<img width="1201" height="703" alt="image" src="https://github.com/user-attachments/assets/e86d586b-c020-4002-bc97-33333329849b" />

Changing any element in the weight matrix does affect the bias matrix. Therefore, it's derivative with respect to some pq element in the weight matrix is always 0. Furthermore, notice that whenever i ≠ p the partial derivative is zero. This is becuase changing Wₚq does not change any Wᵢₖ unless i = p and q = k. In other words, changing a p-qth element in the weight matrix can only create change in elements in the output matrix Z if those elements depend on the p-qth weight matrix element. Therefore, the summation over the rows of Z collapses. Lastly, when k ≠ q, ∂(Wᵢₖ Aₖⱼ)/∂Wₚq is 0. Therefore we get 

<img width="1187" height="592" alt="image" src="https://github.com/user-attachments/assets/e445f586-48cc-4a7c-94f5-9f652b5d6eac" />

Notice that the q-jth element of the Aˡ matrix is the same as the j-qth element of the Aˡ transposed matrix (recall that a transpose is just the flipping of the row and column coordinates of a matrix). So:

<img width="1012" height="330" alt="image" src="https://github.com/user-attachments/assets/e4d95214-ffce-43ff-96d1-76ba01125dcf" />

This, in fact, is the vector dot product for a specific row and column of 2 matrices! Since p and q are arbritrary indexes, this equation can be represented through matrix multiplication. We can then put this equation into matrix form giving: 

<img width="1023" height="369" alt="image" src="https://github.com/user-attachments/assets/98ffcba2-88bc-4c94-8b91-ddd8a1a1f069" />

Note that this calculation is for an arbritrary layer l, and so I have left ∂C/∂Z as a variable since l is not necessarily the hidden layer closest to the output layer.

## ∂C/∂A for Inner Layers (Eg. L-1)
Calculating this derivative is very similar to calculating ∂Z/∂W for some layer l, and so I think it would be a good excercise. Just follow all the steps used in the ∂C/∂W derivation, use the commutativity of multiplication, and matrix transposition to represent the p-qth element of the resulting matrix.

Your answer should be:

<img width="1009" height="283" alt="image" src="https://github.com/user-attachments/assets/4a5e866e-36ac-4d00-a239-e9dc9033279b" />

## ∂C/∂b  
Similarly to ∂C/∂W, we can use the total differential to find ∂C/∂b:

<img width="1114" height="536" alt="image" src="https://github.com/user-attachments/assets/00a9a5db-2260-4cfe-87e8-f4f95d641bcb" />

Again, notice that when i ≠ p, that term in the summation is 0 because ∂Zᵢⱼ/∂bₚ is 0. Changing any element p in the original bias vector at layer l will not affect the weight matrix at layer l since the weight matrix is also an independent variable. It also won't affect the activation matrix of layer l - 1 since it acts as an independent variable at each layer l. Therefore we have:

<img width="771" height="474" alt="image" src="https://github.com/user-attachments/assets/d58cd831-e461-4762-a35e-a7589c9d2758" />

It is possible to write this equation in matrix form. However, I believe the notation is a bit confusing and so I have omitted it in this documentation. This equation is all that is needed to code the derivative of cost with respect to the bias matrix at some layer l.

## The Intuition Behind Back Propagation of the Cost Gradient
A question that you may have when going through these calculations is how we can find the partial derivative of cost with respect to the weight matrices and bias matrices at deeper levels. Using the definition of the total differential and the chain rule tree provided in the above sections to find ∂C/∂W^[l-1], we would get:

<img width="1067" height="241" alt="image" src="https://github.com/user-attachments/assets/a883431e-9f12-4298-a5e2-490cae103f3e" />

Now this looks VERY confusing and, if the equation is viewed as a whole, it is indeed complex. However, focus on the annotation in the equation. The first 2 terms in the equation are something that we have already solved for individually, followed by another partial derivative that we have also solved for individually. This is the intuition behind backwards propagation. Instead of viewing the above equation all at once, we must break the equation down into layers, starting from the hidden layer that is closest to the output layer. Since we are using arbritrary indexes for every matrix in the equation, the techniques with the total differential of cost, demonstrated in the above sections, still apply. 

As you may have deduced, backward propagation is an iterative process and so it is fairly simple to implement in code. Calculate the partial derivative at individual layers as discussed above and store it in a variable (the convention is to store it as d[*name of component matrix*]. Eg. dZ would represent ∂C/∂Z). Then, calculate further gradient components and multiply this value by the stored variable. Store this value in another variable, and so on until you reach the hidden layer closest to the input layer.

# Conclusion
If you are not me from the future, I hope this README file was of help in understanding neural networks. I want this project to be a 1 stop resource to learning about neural networks so that others would not have to spend as much time as I did to understand neural networks. For more details about anything I discussed here, you can check out the sources that I have listed at the start of the file. 

Thank you for reading!
