#By: Hussein Shata
#Feel free to change any of the 4 functions below to see its animated behavior

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def binary_step(x):
    return np.where(x >= 0, 1, 0)

def tanh(x):
    return np.tanh(x)
################################

def plot_sigmoid(ax):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = sigmoid(x_vals)
    ax.plot(x_vals, y_vals, label="Sigmoid function", color='blue')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Input Sum',fontsize=10)
    ax.set_ylabel('Output',fontsize=10)
    ax.set_title('Sigmoid',fontsize=12)
    ax.legend(loc='lower right')
    ax.text(0.8, 0.2, r"$\sigma(x) = \frac{1}{1 + e^{-x}}$", transform=ax.transAxes, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    return ax

def plot_relu(ax):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = relu(x_vals)
    ax.plot(x_vals, y_vals, label="ReLU function", color='blue')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1, 11)
    ax.set_xlabel('Input Sum', fontsize=10)
    ax.set_ylabel('Output', fontsize=10)
    ax.set_title('ReLU', fontsize=12)
    ax.legend(loc='upper right')
    ax.text(0.25, 0.25, r"$ReLU(x) = \max(0, x)$", transform=ax.transAxes, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    return ax

def plot_binary_step(ax):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = binary_step(x_vals)
    ax.plot(x_vals, y_vals, label="Binary Step function", color='blue')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Input Sum',fontsize=10)
    ax.set_ylabel('Output',fontsize=10)
    ax.set_title('Binary Step',fontsize=12)
    ax.legend(loc='upper right')
    equation = "φ(x) = 1 if x ≥ 0\nφ(x)= 0 if x < 0"
    ax.text(0.75, 0.2, equation, transform=ax.transAxes, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    return ax

def plot_tanh(ax):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = tanh(x_vals)
    ax.plot(x_vals, y_vals, label="Tanh function", color='blue')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Input Sum',fontsize=10)
    ax.set_ylabel('Output',fontsize=10)
    ax.set_title('Tanh',fontsize=12)
    ax.legend(loc='lower right')
    ax.text(0.75, 0.2, r"$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$", transform=ax.transAxes, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    return ax

fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 
plt.subplots_adjust(wspace=0.25, hspace=0.25) 

plot_sigmoid(axs[0, 0])
plot_relu(axs[0, 1])
plot_binary_step(axs[1, 0])
plot_tanh(axs[1, 1])

points = []
texts = []
for ax in axs.flat:
    point, = ax.plot([], [], 'ro', markersize=8) #red dot
    points.append(point)
    text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=8)
    texts.append(text)

def update(frame):
    input_sum = -10 + frame * 0.2
    for point, text, func in zip(points, texts, [sigmoid, relu, binary_step, tanh]):
        output = func(input_sum)
        point.set_data(input_sum, output)  #red dot position
        text.set_text(f'Sum, x = {input_sum:.2f},\nOutput = {output:.2f}')
    return points + texts

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True) 
ani.save('activation_functions.gif', writer='pillow')

plt.tight_layout()
plt.show()
