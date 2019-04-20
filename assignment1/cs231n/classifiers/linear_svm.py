import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

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

    dW = np.zeros(W.shape)  # initialize the gradient as zero
    llsss = np.zeros(len(X) * W.shape[1]).reshape(len(X), W.shape[1])
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                llsss[i, j] = margin
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.

    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    (num_train, num_features) = X.shape
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    # extract the w corresponding to each  y
    svmScores = X.dot(W)
    svmCorrectScore = svmScores[np.arange(num_train), y]

    # we need to add axis to the current vector to be able to add it  to a matrix
    # http://www.scipy-lectures.org/intro/numpy/operations.html
    svmCorrectScore = svmCorrectScore[:, np.newaxis]
    li = svmScores - svmCorrectScore + 1
    li[li < 0] = 0
    # this  includes 1 for the cases of the correct class while it should be considered 0
    # we can  add li[li==1]=0 but for future use of li in gradient section we won't do that

    li[li == 1] = 0
    loss = np.sum(li)

    # this sum includes 1 for the cases of the correct class while it should be considered 0
    # hence we subtract these 1s

    # loss -=  num_train
    loss /= num_train
    loss += reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################



    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    np.set_printoptions(precision=3)
    mask = np.zeros(li.shape)
    mask[li > 0] = 1
    row_sum = np.sum(mask, axis=1)
    mask[np.arange(num_train), y] = -row_sum.T
    dW = np.dot(X.T, mask)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
