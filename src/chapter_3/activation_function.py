import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)

# plt.show()
plt.savefig('step_function.png')

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.figure() # 初期化
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.savefig('sigmoid.png')

# ReLU関数
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.figure() # 初期化
plt.plot(x, y)
plt.ylim(-0.1, 5.1)
plt.savefig('relu.png')
