# Task01_01

import numpy as np

arr = np.random.exponential(scale=1.0, size=(4, 4))

print(arr)

# Task01_02
import numpy as np
import matplotlib.pyplot as plt

exp_data = np.random.exponential(scale=1.0, size=(100000, 1))  # Exponential
uni_data = np.random.rand(100000, 1)                           # Uniform [0,1)
norm_data = np.random.randn(100000, 1)                         # Normal (mean=0, std=1)

# histogram
plt.hist(exp_data, density=True, bins=200, histtype="step", color="green", label="exponential")
plt.hist(uni_data, density=True, bins=200, histtype="step", color="blue", label="uniform")
plt.hist(norm_data, density=True, bins=200, histtype="step", color="red", label="normal")

# Adjust axis for better view
plt.axis([-2.5, 6, 0, 1.2])   # Show left tail for normal, right tail for exponential

# Labels & legend
plt.legend(loc="upper right")
plt.title("Random Distributions (Exponential, Uniform, Normal)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# Task01_03
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# Create a meshgrid for X and Y
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Define Z = X^2 + Y^2
Z = X**2 + Y**2

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

# Labels
ax.set_title("3D Surface Plot: Z = X² + Y²")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

# Task01_04
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Pokémon dataset (adjust if your tutorial notebook loads differently)
df = sns.load_dataset("pokemon")

# Select the required features
features = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
data = df[features].dropna()

# Pearson correlation
pearson_corr = data.corr(method="pearson")

# Spearman correlation
spearman_corr = data.corr(method="spearman")

# Plot Pearson heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

# Plot Spearman heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Spearman Rank Correlation Heatmap")
plt.show()