import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    for i in xrange(num_train):
        f = np.dot(X[i, :], W)
        f -= np.max(f)
        p = np.exp(f) / np.sum(np.exp(f))
        loss -= np.log(p[y[i]])

        coeff = np.zeros(len(f))
        coeff = p
        coeff[y[i]] -= 1
        coeff = coeff[:, np.newaxis]
        coeff /= num_train
        xx = X[i]
        xx = xx[:, np.newaxis]
        dW += xx * coeff.T

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    print("losssssssssssssssssssssssssssssssss", loss, loss1)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # scores = np.dot(X,W)
    # scores = np.exp(scores)
    # p = -np.log( scores[range(len(scores)),y]/np.sum(scores,axis=1))
    # loss1 = np.sum(p)/len(p) +  0.5 * reg * np.sum(W*W)


    f = np.dot(X, W)
    f -= np.max(f, axis=0)
    f = np.exp(f)
    p = f / np.sum(f, axis=0, keepdims=True)

    # softmax function
    loss = np.sum(- np.log(p[np.arange(X.shape[0]), y]))
    loss /= X.shape[0]
    loss += 0.5 * reg * np.sum(W * W)

    dscores = p
    dscores[np.arange(dscores.shape[0]), y] -= 1
    dscores /= dscores.shape[0]
    dW = np.dot(X.T, dscores)

    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
