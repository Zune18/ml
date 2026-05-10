import math

# dataset
x_raw = [10, 20, 30, 40, 50]
y = [1, 0, 0, 1, 1]

# standardization - subtraction by mean & dividing by SD
n = len(x_raw)
mean_x = sum(x_raw) / n
variance = sum((i - mean_x) ** 2 for i in x_raw) / n
std_dev_x = math.sqrt(variance)

x = [(i - mean_x) / std_dev_x for i in x_raw]

m = 0
b = 0
learning_rate = 0.1
epochs = 1000

def sigmoid(z):
    # clipping z to avoid overflow in math.exp
    z = max(min(z, 500), -500)
    return 1 / (1 + math.exp(-z))

for epoch in range(epochs):
    dm = 0
    db = 0
    loss = 0

    for i in range(n):
        z = m * x[i] + b
        y_pred = sigmoid(z)
        
        # clip y_pred to avoid log(0) errors
        y_pred = max(min(y_pred, 0.9999999), 0.0000001)

        error = y_pred - y[i]

        # gradients
        dm += error * x[i]
        db += error

        # binary cross entropy loss
        loss += -(y[i] * math.log(y_pred) + (1 - y[i]) * math.log(1 - y_pred))

    dm /= n
    db /= n
    loss /= n

    m -= learning_rate * dm
    b -= learning_rate * db

    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch}, Loss: {loss:.4f}")

# print(f"\nFinal Parameters: m = {m:.4f}, b = {b:.4f}\n")

test_val = 30

# Imp: Use the MEAN and STD_DEV from the TRAINING data
test_x_scaled = (test_val - mean_x) / std_dev_x

z = m * test_x_scaled + b
y_prob = sigmoid(z)

print(f"Test Value: {test_val}")
print(f"Standardized Test Value: {test_x_scaled:.4f}")
print(f"Linear Score (z): {z:.4f}")
print(f"Probability (y_pred): {y_prob:.4f}")