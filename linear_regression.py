x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# normalize
x = [i / max(x) for i in x]
y = [i / max(y) for i in y]

m = 0
b = 0

learning_rate = 0.1
epochs = 500
n = len(x)

for epoch in range(epochs):
    dm = 0
    db = 0
    total_error = 0

    for i in range(n):
        y_pred = m * x[i] + b
        error = y[i] - y_pred

        total_error += error ** 2

        dm += -2 * x[i] * error
        db += -2 * error

    dm /= n
    db /= n

    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch}, Gradients: dm = {dm:.2f} db = {db:.2f}")

    m -= learning_rate * dm
    b -= learning_rate * db

    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch}, Loss: {total_error/n:.2f}")

print(f"\nFinal: y = {m:.2f}x + {b:.2f}\n")