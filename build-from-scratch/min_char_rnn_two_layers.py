"""
Minimal character-based language model learning with RNNs.
Taken from Andrej Karpathy's min-char-rnn:
    https://gist.github.com/karpathy/d4dee566867f8291f086
The companion blog post is:
  https://eli.thegreenplace.net/2018/understanding-how-to-implement-a-character-based-rnn-language-model/.
Modified in various ways for better introspection / customization, Python 3
compatibility and added comments. I tried to retain the overall structure of
this code almost identical to the original.
To run, learning a char-based language model from some text:
    $ python min-char-rnn.py <text file>
----
Original license/copyright blurb:
Minimal character-level Vanilla RNN model.
Written by Andrej Karpathy (@karpathy)
BSD License
"""
from __future__ import print_function

import numpy as np
import sys

# Make it possible to provide input file as a command-line argument; input.txt
# is still the default.
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'input.txt'

with open(filename, 'r') as f:
    data = f.read()

# All unique characters / entities in the data set.
chars = list(set(data))
chars.sort()
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

# Each character in the vocabulary gets a unique integer index assigned, in the
# half-open interval [0:N). These indices are useful to create one-hot encoded
# vectors that represent characters in numerical computations.
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
print('char_to_ix', char_to_ix)
print('ix_to_char', ix_to_char)

# Hyperparameters
hidden_size = 512  # size of hidden layer of neurons
seq_length = 16  # number of steps to unroll the RNN for
learning_rate = 1e-2

# Stop when processed this much data
MAX_DATA = 100000
MAX_ITER = 200000

# Model parameters/weights -- these are shared among all steps. Weights
# initialized randomly; biases initialized to 0.
# Inputs are characters one-hot encoded in a vocab-sized vector.
# Dimensions: H = hidden_size, V = vocab_size
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh1 = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden in layer one
Whh2 = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden in layer two
Wh1h2 = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden from layer one to layer two
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh1 = np.zeros((hidden_size, 1))  # hidden bias
bh2 = np.zeros((hidden_size, 1)) # hidden bias from layer one to layer two
by = np.zeros((vocab_size, 1))  # output bias


def lossFun(inputs, targets, hprev):
    """Runs forward and backward passes through the RNN.
  inputs, targets: Lists of integers. For some i, inputs[i] is the input
                   character (encoded as an index into the ix_to_char map) and
                   targets[i] is the corresponding next character in the
                   training data (similarly encoded).
  hprev: Hx2 array of initial hidden state
  returns: loss, gradients on model parameters, and last hidden state
  """
    # Caches that keep values computed in the forward pass at each time step, to
    # be reused in the backward pass.
    xs, h1s, h2s, ys, ps = {}, {}, {}, {}, {}

    # Initial incoming state.
    h1s[-1] = np.expand_dims(np.copy(hprev[:, 0]), 1)
    h2s[-1] = np.expand_dims(np.copy(hprev[:, 1]), 1)

    loss = 0
    # Forward pass
    for t in range(len(inputs)):
        # Input at time step t is xs[t]. Prepare a one-hot encoded vector of shape
        # (V, 1). inputs[t] is the index where the 1 goes.
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # Compute h1[t], h2[t] from h1[t-1], h2[t-1] and x[t]
        h1s[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh1, h1s[t-1]) + bh1)
        h2s[t] = np.tanh(np.dot(Wh1h2, h1s[t]) + np.dot(Whh2, h2s[t-1]) + bh2)

        # Compute ps[t] - softmax probabilities for output.
        ys[t] = np.dot(Why, h2s[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # Cross-entropy loss for two probability distributions p and q is defined as
        # follows:
        #
        #   xent(q, p) = -Sum q(k)log(p(k))
        #                  k
        #
        # Where k goes over all the possible values of the random variable p and q
        # are defined for.
        # In our case taking q is the "real answer" which is 1-hot encoded; p is the
        # result of softmax (ps). targets[t] has the only index where q is not 0,
        # so the sum simply becomes log of ps at that index.
        loss += -np.log(ps[t][targets[t], 0])

    # Backward pass: compute gradients going backwards.
    # Gradients are initialized to 0s, and every time step contributes to them.
    dWxh, dWhh1, dWhh2, dWh1h2, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh1), np.zeros_like(Whh2), np.zeros_like(Wh1h2), np.zeros_like(Why)
    dbh1, dbh2, dby = np.zeros_like(bh1), np.zeros_like(bh2), np.zeros_like(by)

    # Initialize the incoming gradient of h to zero; this is a safe assumption for
    # a sufficiently long unrolling.
    dh1next = np.zeros_like(h1s[0])
    dh2next = np.zeros_like(h2s[0])

    # The backwards pass iterates over the input sequence backwards.
    for t in reversed(range(len(inputs))):
        # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # Compute gradients for the Why and by parameters.
        dWhy += np.dot(dy, h2s[t].T)
        dby += dy

        # Backprop through the fully-connected layer (Why, by) to h2. Also add up the
        # incoming gradient for h2 from the next cell.
        # Note: proper Jacobian matmul here would be dy.dot(Why), that would give
        # a [1,T] vector. Since we need [T,1] for h, we flip the dot (we could have
        # transposed after everything, too)
        dh2 = np.dot(Why.T, dy) + dh2next
        # Backprop through the tanh in layer two.
        dh2raw = (1 - h2s[t] * h2s[t]) * dh2
        # Compute gradients for the dbh2, dWh1h2, Whh2 parameters.
        dbh2 += dh2raw
        dWh1h2 += np.dot(dh2raw, h1s[t].T)
        dWhh2 += np.dot(dh2raw, h2s[t-1].T)

        # Backprop through the fully-connected layer (Wh1h2, dbh1) to h1. Also add up the
        # incoming gradient for h1 from the next cell.
        dh1 = np.dot(Wh1h2.T, dh2raw) + dh1next
        dh1raw = (1 - h1s[t] * h1s[t]) * dh1
        # Compute
        dbh1 += dh1raw
        dWxh += np.dot(dh1raw, xs[t].T)
        dWhh1 += np.dot(dh1raw, h1s[t-1].T)

        # Backprop the gradient to the incoming h, which will be used in the
        # previous time step.
        dh2next = np.dot(Whh2.T, dh2raw)
        dh1next = np.dot(Whh1.T, dh1raw)

    # Gradient clipping to the range [-5, 5].
    for dparam in [dWxh, dWhh1, dWhh2, dWh1h2, dWhy, dbh1, dbh2, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh1, dWhh2, dWh1h2, dWhy, dbh1, dbh2, dby, np.concatenate((h1s[len(inputs) - 1], h2s[len(inputs)-1]), axis=1)


def sample(h, seed_ix, n):
    """Sample a sequence of integers from the model.
  Runs the RNN in forward mode for n steps; seed_ix is the seed letter for the
  first time step, and h is the memory state. Returns a sequence of letters
  produced by the model (indices).
  """
    # Create a one-hot vector to represent the input.
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    h1 = np.expand_dims(np.copy(h[:, 0]), 1)
    h2 = np.expand_dims(np.copy(h[:, 1]), 1)

    for t in range(n):
        # Run the forward pass only.
        h1 = np.tanh(np.dot(Wxh, x) + np.dot(Whh1, h1) + bh1)
        h2 = np.tanh(np.dot(Wh1h2, h1) + np.dot(Whh2, h2) + bh2)
        y = np.dot(Why, h2) + by
        p = np.exp(y) / np.sum(np.exp(y))

        # Sample from the distribution produced by softmax.
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        # Prepare input for the next cell.
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# Gradient checking (from karpathy's own comment on the gist)
from random import uniform


def gradCheck(inputs, targets, hprev):
    global Wxh, Whh1, Whh2, Wh1h2, Why, bh1, bh2, by
    num_checks, delta = 30, 1e-5
    _, dWxh, dWhh1, dWhh2, dWh1h2, dWhy, dbh1, dbh2, dby, _ = lossFun(inputs, targets, hprev)
    for param, dparam, name in zip([Wxh, Whh1, Whh2, Wh1h2, Why, bh1, bh2, by],
                                   [dWxh, dWhh1, dWhh2, dWh1h2, dWhy, dbh1, dbh2, dby],
                                   ['Wxh', 'Whh1', 'Whh2', 'Wh1h2', 'Why', 'bh', 'bhh', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1)
        print(name)
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less


# This function invokes gradCheck with all the parameters properly set up.
def basicGradCheck():
    inputs = [char_to_ix[ch] for ch in data[:seq_length]]
    targets = [char_to_ix[ch] for ch in data[1:seq_length + 1]]
    hprev = np.zeros((hidden_size, 2))  # reset RNN memory
    gradCheck(inputs, targets, hprev)


# Uncomment this to run a basic gradient check.
# basicGradCheck()

# n is the iteration counter; p is the input sequence pointer, at the beginning
# of each step it points at the sequence in the input that will be used for
# training this iteration.
n, p = 0, 0

# Memory variables for Adagrad.
mWxh, mWhh1, mWhh2, mWh1h2, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh1), np.zeros_like(Whh2), np.zeros_like(Wh1h2), np.zeros_like(Why)
mbh1, mbh2, mby = np.zeros_like(bh1), np.zeros_like(bh2), np.zeros_like(by)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length

while n < MAX_ITER:
    # Prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 2))  # reset RNN memory
        p = 0  # go from start of data

    # In each step we unroll the RNN for seq_length cells, and present it with
    # seq_length inputs and seq_length target outputs to learn.
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # gradCheck(inputs, targets, hprev)
    # break

    # Sample from the model now and then.
    if n % 1000 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    # Forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh1, dWhh2, dWh1h2, dWhy, dbh1, dbh2, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 200 == 0: print('iter %d (p=%d), loss: %f' % (n, p, smooth_loss))

    # Perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh1, Whh2, Wh1h2, Why, bh1, bh2, by],
                                  [dWxh, dWhh1, dWhh2, dWh1h2, dWhy, dbh1, dbh2, dby],
                                  [mWxh, mWhh1, mWhh2, mWh1h2, mWhy, mbh1, mbh2, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1