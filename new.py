import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000               # Number of users
video_types = 4         # Number of video types
alpha = 0.1             # Confidence level
log_term = np.log(1 / alpha)

# Initialize arrays for counting clicks and recommendations
m_k = np.zeros(video_types)  # Clicks for each video type
n_k = np.zeros(video_types)  # Recommendations for each video type
R_k = np.zeros(video_types)  # Revenue for each video type

# Store results for each iteration
m_s_avg = np.zeros((N, video_types))  # Average clicks
n_s_avg = np.zeros((N, video_types))  # Average recommendations
R_s_avg = np.zeros((N, video_types))  # Average revenue

# True probabilities and revenue factors
p_true = [0.2, 0.4, 0.6, 0.65]
a_k = [2, 2, 2, 2]

# Function to calculate Upper Confidence Bound (UCB)
def calc_ucb(n_k, m_k):
    ucb_k = np.zeros(video_types)
    for k in range(video_types):
        if n_k[k] > 0:
            ucb_k[k] = (m_k[k] / n_k[k]) + np.sqrt(log_term / (2 * n_k[k]))
        else:
            ucb_k[k] = 1  # Encourage exploration for untried actions
    return ucb_k

# Number of iterations for simulation
num_iterations = 1000

# Run the simulation
for _ in range(num_iterations):
    # Reset counts for each iteration
    n_k = np.zeros(video_types)
    m_k = np.zeros(video_types)
    R_k = np.zeros(video_types)

    for s in range(N):
        UCB_k = calc_ucb(n_k, m_k)
        chosen_type = np.argmax(UCB_k)  # Select the type with highest UCB
        clicked = np.random.rand() < p_true[chosen_type]  # Simulate click
        n_k[chosen_type] += 1  # Increment recommendation count
        m_k[chosen_type] += clicked  # Increment clicks
        R_k[chosen_type] += a_k[chosen_type] * np.random.rand() * clicked  # Increment revenue

        # Store results for average calculations
        n_s_avg[s] += n_k
        m_s_avg[s] += m_k
        R_s_avg[s] += R_k

# Calculate the average results
m_s_avg /= num_iterations  # Average clicks
n_s_avg /= num_iterations  # Average recommendations
R_s_avg /= num_iterations  # Average revenue
for k in range(video_types):
    for s in range(N):
        if n_s_avg[s][k] == 0 : n_s_avg[s][k] = 1
# Calculate m/n for each video type
m_by_n_avg = np.zeros((N, video_types))
for k in range(video_types):
    m_by_n_avg[:, k] = m_s_avg[:, k] / n_s_avg[:, k]  # Avoid division by zero

# Plot the results
# plt.figure(figsize=(12, 6))
for k in range(video_types):
    plt.plot(R_s_avg[:, k], label=f'Video Type {k+1}')
# plt.xlabel('Number of Users (s)')
# plt.ylabel('Click Rate (m/n)')
# plt.title('Click Rate for Each Video Type as a Function of Users')
# plt.legend()
plt.grid()
plt.show()
# plt.plot(m_by_n_avg[:,0])
# plt.show()
# plt.plot(R_s_avg)