import numpy as np
import matplotlib.pyplot as plt

# Recap



# Optimization
# minimize f(x) = (1+x^2)

x = np.arange(-5, 5, 0.01)
y = 1 + x**2

#2*x = 0
#x = 0

x_random = np.random.randn(100)
y_random = 1 + x_random**2
plt.scatter(x_random, y_random)
x_random[np.argmin(y_random)]

# Data generation
x = np.arange(-5, 5, 0.01)
w_true = 1
b_true = 0
y_true = w_true * x + b_true
y_data = w_true * (x + np.random.randn(len(y_true))) + b_true

plt.scatter(x, y_data, s=1)

def loss(y_model, y_true):
    return np.mean((y_true - y_model)*(y_true - y_model))


# Grid search
loss_matrix = np.zeros((200, 200))
for i, w in enumerate(np.arange(-10, 10, 0.1)):
    for j, b in enumerate(np.arange(-10, 10, 0.1)):
        y_model = w * x + b
        loss_matrix[i, j] = loss(y_model, y_data)
plt.imshow(loss_matrix)


# Gradient Descent
#d(y_data - w * x + b)^2/dw
#2(y_data - w * x + b)*x
#w_est = w_est + residual*data

w_est = 0
l = 0.1
w_history=[]
loss_history=[]
for i in range(10):
    w_history.append(w_est)
    loss_history.append(loss(w_est * x + b, y_data))
    r = y_data - w_est * x + b
    w_est = w_est + l * np.mean(r*x)
plt.plot(w_history, loss_history)