import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define constants for probabilities and revenue bounds for each video type
p = [0.2, 0.4, 0.6, 0.65]  # Probabilities for video types 1, 2, 3, and 4
a = [2, 2, 2, 2]            # Uniform distribution bounds (assumed same for simplicity)
mu_diff = [(p_k - p[3]) * (a_k / 2) for p_k, a_k in zip(p[:3], a[:3])]  # Differences in expected revenues
sigma_diff = np.sqrt(2 / 3)  # Standard deviation difference for all comparisons (assuming uniform distributions)

# Range of N1 values
N1_values = np.linspace(1, 1000, 500)

# Calculate the probability of choosing each wrong video type
wrong_prob = np.zeros_like(N1_values)
for mu in mu_diff:
    wrong_prob += 1 - norm.cdf((mu * np.sqrt(N1_values)) / sigma_diff)

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(N1_values, wrong_prob, label="P(Choosing Wrong Type)", color='b')
plt.xlabel('N1 (Number of Explorations)', fontsize=12)
plt.ylabel('Total Probability of Choosing the Wrong Type', fontsize=12)
plt.title('Total Probability of Choosing the Wrong Type vs. N1', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
