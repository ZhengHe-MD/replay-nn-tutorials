"""
A minimal character-level LSTM model. Written by ZhengHe (@ZhengHe-MD)
This is derived from the following scripts:
- https://gist.github.com/karpathy/d4dee566867f8291f086
- https://github.com/eliben/deep-learning-samples/blob/master/min-char-rnn/min-char-rnn.py
- https://github.com/nicodjimenez/lstm
And you might find the following materials helpful:
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- https://nicodjimenez.github.io/2014/08/08/lstm.html
- http://arxiv.org/abs/1506.00019
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- https://explained.ai/matrix-calculus/index.html

To run:

    $ python min_char_rnn_lstm.py <text file>

----
BSD License
"""
import numpy as np
import sys

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
hidden_size = 50  # size of hidden layer of neurons
seq_length = 16  # number of steps to unroll the RNN for
learning_rate = 1e-1

ub, lb = 0.1, -0.1
# LSTM
Wgs = np.random.randn(seq_length, hidden_size, hidden_size + vocab_size) * (ub - lb) + lb
Wis = np.random.randn(seq_length, hidden_size, hidden_size + vocab_size) * (ub - lb) + lb
Wfs = np.random.randn(seq_length, hidden_size, hidden_size + vocab_size) * (ub - lb) + lb
Wos = np.random.randn(seq_length, hidden_size, hidden_size + vocab_size) * (ub - lb) + lb
bgs = np.zeros((seq_length, hidden_size, 1))
bis = np.zeros((seq_length, hidden_size, 1))
bfs = np.zeros((seq_length, hidden_size, 1))
bos = np.zeros((seq_length, hidden_size, 1))
# Fully-connected
Why = np.random.randn(vocab_size, hidden_size) * (ub - lb) + lb
by = np.zeros((vocab_size, 1))


def lossFun(inputs, targets, hprev, sprev):
    assert len(inputs) == seq_length
    assert len(targets) == seq_length

    xs, hs, ss, ps, ys = {}, {}, {}, {}, {}
    gs, iis, fs, os = {}, {}, {}, {}  # the `iis` here should be `is`, unfortunately `is` is a keyword in python
    # Initial incoming state.
    hs[-1] = np.copy(hprev)
    ss[-1] = np.copy(sprev)

    loss = 0
    # Forward pass
    for t in range(seq_length):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        xc = np.vstack((xs[t], hs[t - 1]))

        gs[t] = np.tanh(np.dot(Wgs[t], xc) + bgs[t])
        iis[t] = sigmoid(np.dot(Wis[t], xc) + bis[t])
        fs[t] = sigmoid(np.dot(Wfs[t], xc) + bfs[t])
        os[t] = sigmoid(np.dot(Wos[t], xc) + bos[t])
        ss[t] = gs[t] * iis[t] + ss[t - 1] * fs[t]
        hs[t] = ss[t] * os[t]

        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = softmax(ys[t])
        loss += -np.log(ps[t][targets[t], 0])

    # Backward pass
    dWgs, dWis, dWfs, dWos = np.zeros_like(Wgs), np.zeros_like(Wis), np.zeros_like(Wfs), np.zeros_like(Wos)
    dbgs, dbis, dbfs, dbos = np.zeros_like(bgs), np.zeros_like(bis), np.zeros_like(bfs), np.zeros_like(bos)
    dWhy, dby = np.zeros_like(Why), np.zeros_like(by)

    dh_next = np.zeros_like(hprev)
    ds_next = np.zeros_like(sprev)

    for t in reversed(range(seq_length)):
        # Backprop through the gradients of loss and softmax
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        dh = np.dot(Why.T, dy) + dh_next

        ds = os[t] * dh + ds_next
        do = ss[t] * dh
        di = gs[t] * ds
        dg = iis[t] * ds
        df = ss[t - 1] * ds

        di_input = sigmoid_derivative(iis[t]) * di
        df_input = sigmoid_derivative(fs[t]) * df
        do_input = sigmoid_derivative(os[t]) * do
        dg_input = tanh_derivative(gs[t]) * dg

        xc = np.vstack((xs[t], hs[t - 1]))
        dWis[t] = np.outer(di_input, xc)
        dWfs[t] = np.outer(df_input, xc)
        dWos[t] = np.outer(do_input, xc)
        dWgs[t] = np.outer(dg_input, xc)
        dbis[t] = di_input
        dbfs[t] = df_input
        dbos[t] = do_input
        dbgs[t] = dg_input

        dxc = np.zeros_like(xc)
        dxc += np.dot(Wis[t].T, di_input)
        dxc += np.dot(Wfs[t].T, df_input)
        dxc += np.dot(Wos[t].T, do_input)
        dxc += np.dot(Wgs[t].T, dg_input)

        ds_next = ds * fs[t]
        dh_next = dxc[vocab_size:]

    for dparam in [dWgs, dWis, dWfs, dWos, dbgs, dbis, dbfs, dbos, dWhy, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWgs, dWis, dWfs, dWos, dbgs, dbis, dbfs, dbos, dWhy, dby, hs[seq_length - 1], ss[seq_length - 1]


def sample(h, s, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        xc = np.vstack((x, h))
        tt = t % seq_length
        g = np.tanh(np.dot(Wgs[tt], xc) + bgs[tt])
        i = sigmoid(np.dot(Wis[tt], xc) + bis[tt])
        f = sigmoid(np.dot(Wfs[tt], xc) + bfs[tt])
        o = sigmoid(np.dot(Wos[tt], xc) + bos[tt])
        s = g * i + s * f
        h = s * o
        y = np.dot(Why, h) + by
        p = softmax(y)

        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1. - x)


def tanh_derivative(x):
    return 1. - x * x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def gradCheck(inputs, targets, hprev, sprev):
    from random import uniform

    global Wgs, Wis, Wfs, Wos, bgs, bis, bfs, bos, Why, by
    num_checks, delta = 10, 1e-4
    loss, dWgs, dWis, dWfs, dWos, dbgs, dbis, dbfs, dbos, dWhy, dby, _, _ = lossFun(inputs, targets, hprev, sprev)
    for param, dparam, name in zip([Wgs, Wis, Wfs, Wos, bgs, bis, bfs, bos, Why, by],
                                   [dWgs, dWis, dWfs, dWos, dbgs, dbis, dbfs, dbos, dWhy, dby],
                                   ['Wgs', 'Wis', 'Wfs', 'Wos', 'bgs', 'bis', 'bfs', 'bos', 'Why', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, f"Error dims don't match {s0} and {s1}."
        print(name)
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev, sprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev, sprev)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))


def basicGradCheck():
    inputs = [char_to_ix[ch] for ch in data[:seq_length]]
    targets = [char_to_ix[ch] for ch in data[1:seq_length + 1]]
    hprev = np.zeros((hidden_size, 1))  # reset RNN memory
    sprev = np.zeros((hidden_size, 1))
    gradCheck(inputs, targets, hprev, sprev)


# Uncomment this to run a basic gradient check.
# basicGradCheck()

n, p = 0, 0

mWgs, mWis, mWfs, mWos = np.zeros_like(Wgs), np.zeros_like(Wis), np.zeros_like(Wfs), np.zeros_like(Wos)
mbgs, mbis, mbfs, mbos = np.zeros_like(bgs), np.zeros_like(bis), np.zeros_like(bfs), np.zeros_like(bos)
mWhy, mby = np.zeros_like(Why), np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length

MAX_DATA = 1000000

while p < MAX_DATA:
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        sprev = np.zeros((hidden_size, 1))
        p = 0  # go from start of data

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    if n % 1000 == 0:
        sample_ix = sample(hprev, sprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    loss, dWgs, dWis, dWfs, dWos, dbgs, dbis, dbfs, dbos, dWhy, dby, hprev, sprev = lossFun(inputs, targets, hprev, sprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 200 == 0: print('iter %d (p=%d), loss: %f' % (n, p, smooth_loss))

    for param, dparam, mem in zip([Wgs, Wis, Wfs, Wos, bgs, bis, bfs, bos, Why, by],
                                  [dWgs, dWis, dWfs, dWos, dbgs, dbis, dbfs, dbos, dWhy, dby],
                                  [mWgs, mWis, mWfs, mWos, mbgs, mbis, mbfs, mbos, mWhy, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1
