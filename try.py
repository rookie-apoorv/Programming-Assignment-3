import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
N = 10000
video_types = 4
alpha = 0.1
log_term = np.log(1 / alpha)
m_k = np.zeros(video_types)
n_k = np.zeros(video_types)
R_k = np.zeros(video_types)
m_s = np.zeros((N,video_types))
n_s = np.zeros((N,video_types))
R_s = np.zeros((N,video_types))
UCB_K = np.zeros(video_types)
p_true = [0.2,0.4,0.6,0.65]
a_k = [2,2,2,2]
def calc_ucb(n_k,m_k,s) :
    ucb_k = np.zeros(video_types)
    for k in range(video_types):
        if n_k[k] > 0:
            ucb_k = (m_k[k]/n_k[k]) + np.sqrt(log_term / 2 * n_k[k])
        else:
            ucb_k[k] = 1
    return ucb_k


for iter in range(1000):
    for s in range(10000):
        UCB_k = calc_ucb(n_k,m_k,s)

        chosen_type = np.argmax(UCB_K)

        clicked = np.random.rand() < p_true[chosen_type]

        n_k[chosen_type] += 1
        m_k[chosen_type] += clicked
        R_k[chosen_type] += a_k[chosen_type] * np.random.rand() * clicked
        for i in [0,1,2,3]:
            n_s[s][video_types] += n_k[video_types]
            m_s[s][video_types] += m_k[video_types]
            R_s[s][video_types] += R_k[video_types]

m_by_n_k = [m_s[s][0]/n_s[s][0] for s in range(10000)] 
plt.scatter([i for i in range(10000)] , m_by_n_k)
plt.show()
            

