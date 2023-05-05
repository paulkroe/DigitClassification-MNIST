

# MNIST Digit Classification and Deep neural networks.
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) explains core concepts of Neural Networks and Deep Learning.
This project aims at summarizing the book chapter by chapter. After crafting this summary the goal is to build a Neural Network from scratch, implementing the ideas presented in the book, without looking at the material again.

# Neural Networks and Deep Learning
## Chapter 1: Using neural nets to recognize handwritten digits


### Useful functions:

1) The sigmoid activation function: 
$\sigma(z) = \frac{1}{1+e^{-z}}$
More explicitly, the output of a sigmoid neuron with inputs $x_1,...,x_n$ and weights $w_1,...,w_n$ and a bias b is given as:
$$\frac{1}{1+\exp(-\sum_{i = 1}^{n}w_ix_i-b)}$$
Derivative of the sigmoid function
$$\sigma '(z) = \sigma(z) (1-\sigma(z))$$

2) Square mean error: $a$ activation in the output layer, $y(x)$ desired output, $x$ input:
$$C(w,b) = \frac{1}{2n}\sum_{x}||y(x) -a||^2$$

### Forward pass:
Using this we have:
$$a_{j}^l = \sigma(\sum_k w_{jk}^{l}a_{k}^{l-1}+b_{j}^l)$$
Vectorizing $\sigma$ we get:
$$a^{l} = \sigma( w^{l}a^{l-1}+b^{l})$$

### Learning using gradient descent:

High level idea: we look at a cost function $C(w,b)$, dependent on the weights $w$ and biases $b$ of the give neural network (The function doesn't depend on an input $x$, since it measures the cost/loss over all the inputs). The cost function returns a scalar value $c$. Intuitively, $c$ is small if the model made a good predication (on all the inputs) and $c$ is big if the prediction was bad (on all the inputs). This allows us to reframe the learning problem: We want to adjust the weights and biases of the network such that $C(w,b)$ becomes small. To achieve this, we are interested in the partial derivatives: 
$\frac{\partial C}{w_i}$ and $\frac{\partial C}{b_i}$. For that denote $v$ the input to the network. We then have the relation:
$$\Delta C \approx \nabla C \cdot \Delta v$$
(where $\nabla = (\frac{\partial C}{v_1},...,\frac{\partial C}{v_n})^T$)

Now if we choose $\Delta v = - \eta \nabla C$ we have (approximatively) $\Delta C = -\eta \Delta C \cdot \Delta C$. (This equation makes sense since $C$ is real valued) As $\Delta C \cdot \Delta C$ is guaranteed to be positive, we may indeed hope that setting $v \rightarrow v' = v - \nabla C$ will lead to $C(v) \geq C(v')$. Doing this repeatedly, we hope to find a (local) minimum $v$.
Using this approach in the context of machine learning yields the update rule (component wise): $$w_l \rightarrow w_l' = w_l - \eta \frac{\partial C}{\partial w_l}$$
$$b_k \rightarrow b_k' = b_k - \eta \frac{\partial C}{\partial b_k}$$
Since it is computationally unfeasible to calculate these partial derivatives analytically we will use an algorithm called backpropagation to calculate these.

### Stochastic gradient decent:
In practice evaluating the cost function $C$ is very expensive (consider for example the mean square error mentioned above, evaluating $C(w,b)$ involves computing the model's output for all $(x,y)$ in the testdata.) To mitigate this, we approximate the cost function by just evaluating a few data points (x,y) and assume that this value approximates the true value of the cost function. How many data points we consider is governed by the batch size parameter.
$$ \frac{\sum_{j = 1}^{m}\nabla C_{x_j}}{m} \approx \frac{\sum_{j = 1}^{n}\nabla C_{x_j}}{n} = \nabla C$$

## Chapter II: How the backpropagation algorithm works

### Notation:
The book uses $w_{jk}^l$ to refer to weight connecting the $k-th$ neuron in the $l-1-th$ layer to the $j-th$ neuron in the $l-th$ layer. 
Similarly, we use $b_{j}^l$ to refer to the $j-th$ bias in the $l-th$ layer, analogously we use $a_{k}^l$. 

### Backpropagation idea:

Note that if we change the activation $z_{j}^{l}$ by $\Delta z_{j}^l$ then the cost function will change according to $\frac{\partial C}{\partial z_{j}^{l}} \Delta z_{j}^{l}$

Define the error of the $j-th$ neuron in the $l-th$ layer.

$$\delta_{j}^{l} = \frac{\partial C}{\partial z_{j}^{l}}$$

The following four formulas allow us to connect the vector $\delta^{l}$ to the quantities of interest.
### Assumptions:
Firstly, we need that the cost function C and be written as an average:
$$C = \frac{1}{n} \sum_x C_x$$
We need this, since backpropagation will allow us to compute $\frac{\partial C_x}{\partial w}$ and $\frac{\partial C_x}{\partial b}$.

Secondly, we need that the cost function can be written as a function of the neural network in concern.


### Four formulas of backpropagation:
1) For the output layer L we have:
    $$\delta_{j}^{L} = \frac{\partial C}{\partial a_{j}^{L}} \sigma '(z_{j}^{L})$$
    Using $*$ as the pointwise multiplication:
    $$\delta^{L} = \nabla C_a * \sigma '(z^{L})$$

2) Equation connecting the error of the $l-th$ layer to the $l+1-th$ layer:
$$\delta^{l} = ((w^{l+1})^T\delta^{l+1}) * \sigma '(z^{l})$$

3) Error and bias:
$$\frac{\partial C}{\partial b_{j}^{l}} = \delta_{l}^{l}$$

4) Error and weight:
$$\frac{\partial C}{\partial w_{jk}^{l}} = a_{k}^{l-1}\delta_{j}^{l}$$

## Chapter III: Improving the way neural networks learn:
### Cross entropy cost function:
We use the cross entropy function because it mitigates the learning slowdown in the last layer, since its derivative cancels out the derivative of the activation function.

$$C(y,a) = \frac{-1}{n}\sum_x \sum_j [y_j ln(a_j) + (1-y_j) ln(1-a_j)]$$

Using this, the activations in the last layer become:
$$\partial \frac{C}{\partial b_{j}^{L}} = a_{j}^{L} - y_{j}$$
and
$$\partial \frac{C}{\partial w_{jk}^{L}} = \frac{1}{n}\sum_x a_{k}^{L-1}(a_{j}^{L}-y_{j})$$

The term $\sigma ' (z_{j}^{L})$ vanishes. Thus, the learning doesn't slow down (in the last layer).

(Note: This function can be approximated when considering not all inputs x, but only a batch. The function is always positive (since $y_j$ and $a_j$ are assumed to be between 0 and 1)--> be careful when using with ReLUs.)

# Softmax function:
Another way to address learning slowdown in the last layer is using a softmax function.
Instead of using the same activation function as in the rest of the network, one can use a softmax function in the last layer. Similar to the other layers the softmax layer starts by computing the weighted inputs $z_{j}^{L} = \sum_k w_{jk}^{L}a_{k}^{L-1}b_{j}^{L}$. These Inputs are the used in the softmax activation:
$$a_{j}^{L} = \frac{e^{z_{j}^{L}}}{\sum_{k} e^{z_{k}^{L}}}$$ 
By design $\sum_j a_{j}^{L} = 1$. Thus is makes sense to think of the activation of the last layer as an probability distribution.

To use this to address the learning slowdown problem, we define the negative log-likelihood cost function.
$$C=-ln(a_{y}^{L})$$
Where $a_{y}^{L}$ denotes the neuron corresponding to the desired output.
Using that cost function we obtain:
$$\partial \frac{C}{\partial b_{j}^{L}} = a_{j}^{L} - y_{j}$$
and
$$\partial \frac{C}{\partial w_{jk}^{L}} = a_{k}^{L-1}(a_{j}^{L}-y_{j})$$
This a very similar expression to the one that is obtained by using the cross entropy function with sigmoid neurons. The only difference is that it doesn't average over all training examples. So in some sense using cross entropy loss with sigmoids is the same as using softmax and negative log-likelihood.  
### Overfitting and regularization:

### Weight initialization:

### Choice hyperparameters: