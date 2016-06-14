import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z = np.dot(data,W1) + b1
    a = sigmoid(z)
    z2 = np.dot(a,W2) + b2
    a2 = softmax(z2)
    # correct = np.argmax(labels)
    cost = -np.sum(np.log(a2) * labels)
    # print "sizes: "
    # print cost
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    d1 = a2 - labels #Dy x 1
    d2 = np.dot(d1,W2.T) # 1
    d3 = np.multiply(d2,sigmoid_grad(a))
    # print "sizes: " + str(d3.shape)
    gradW2 = np.dot(a.T,d1)
    gradb2 = np.sum(d1,axis=0)
    gradW1 = np.dot(data.T,d3)
    gradb1 = np.sum(d3,axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    # grad = 1

    return cost, grad
    # return cost

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    # forward_backward_prop(data, labels, params, dimensions)

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
