import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# inputs
x1 = 8
x2 = 90

# hidden neuron 1 weights
w1 = 0.5
w2 = 0.1
b1 = -5

# hidden neuron 2 weights
w3 = 0.9
w4 = 0.2
b2 = -10

# output neuron weights
w5 = 0.8
w6 = 0.4
b3 = -2

# hidden layer
h1 = relu((x1 * w1) + (x2 * w2) + b1)
h2 = relu((x1 * w3) + (x2 * w4) + b2)

# output layer
output = sigmoid((h1 * w5) + (h2 * w6) + b3)

print("Hidden neuron 1:", h1)
print("Hidden neuron 2:", h2)
print("Final probability:", output)