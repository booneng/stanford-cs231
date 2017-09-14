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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
     scores = np.dot(X[i], W)
     scores = scores - np.max(scores)
     softmax_scores = np.exp(scores)/np.sum(np.exp(scores))
     loss += - np.log(softmax_scores[y[i]])
     zeros = np.zeros(num_classes)
     zeros[y[i]] = 1
     dscore = softmax_scores - zeros
     dscore = dscore.reshape(num_classes, 1)
     Xn = X[i].reshape(X[i].shape[0], 1)
     dscore = np.dot(dscore, Xn.T)
     dW += dscore.T
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW /num_train + reg * W
    
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_scores = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(-1,1)
  loss = np.sum(-np.log(softmax_scores[range(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) 
  y_matrix = np.zeros((num_train, num_classes))
  softmax_scores[range(num_train), y] -= 1
  dW = np.dot(X.T, softmax_scores)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

