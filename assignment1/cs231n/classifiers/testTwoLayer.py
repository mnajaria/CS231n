import pandas as pd
import numpy as np

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

np.random.seed(1)
x = 10 * np.random.randn(num_inputs, input_size)
y = np.array([0, 1, 2, 2, 1])

eps = 10e-1
reg = 1e-4
w1 = eps * np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
w2 = eps * np.random.randn(hidden_size, num_classes)
b2 = np.zeros(num_classes)

layer1 = np.dot(x, w1) + b1
layer1[layer1 < 0] = 0

scores = np.dot(layer1, w2) + b2
ex = np.exp(scores - np.max(scores, axis=0))
p2 = ex / np.sum(ex, axis=1).reshape(ex.shape[0], 1)
loss = np.sum(-np.log(p2[range(p2.shape[0]), y]))
loss /= p2.shape[0]
loss += 0.5 * reg * (np.sum(w2*w2)+ np.sum(w1*w1))

p2[range(p2.shape[0]), y] -= 1
dw2 = np.dot(layer1.T, p2)
dw2 /= dw2.shape[0]
dw2 += reg * w2
db2 = np.sum(p2, axis=0)

dl = np.dot(w2, p2.T)
xx = layer1
xx[xx > 0] = 1
dh = dl*xx.T
dw1 = x.T.dot(dh.T) + reg * w1

print(dw1)



import matplotlib.pyplot as plt
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
drawFig = False
if drawFig:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, [input_size, hidden_size, num_classes])
    fig.savefig('nn.png')
