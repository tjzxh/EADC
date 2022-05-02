import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
from simulation_env import Env
from ZMX_RL_MLP import DDPG

# Test the trained model
TTC_threshold = 4
beta = 100
env = Env(TTC_threshold)
s_dim = env.n_features
a_dim = env.n_actions
a_bound = env.action_Bound
# Load the trained NN model
base_name = f'origin_4.001_NOCA'
model_file = f'save/model_{base_name}'
ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg.restore(model_file)
# Get model weights
l1_w, l1_b, a_w, a_b = ddpg.print_weight()
# Construct input data
all_input, equi_state = [], []
step = 2
for equi_speed in range(5 * step, 30 * step + 1):
    for possible_gap in range(3 * equi_speed, int(3.5 * equi_speed)):
        all_input.append([possible_gap / step, equi_speed / step, 0])

# Test the trained model
all_input = np.array(all_input)
acce_pred = ddpg.choose_action(all_input)
# Calculate partial differential for stability analysis
y_pred = acce_pred / a_bound
h_s_flag = np.dot(all_input, l1_w) + l1_b
h_s_pd = 1 / (np.exp(-beta * h_s_flag) + 1)
da_ds = a_bound * (1 - y_pred ** 2) * np.dot(l1_w[0, :] * h_s_pd, a_w)
da_dv = a_bound * (1 - y_pred ** 2) * np.dot(l1_w[1, :] * h_s_pd, a_w)
da_dvd = a_bound * (1 - y_pred ** 2) * np.dot(l1_w[2, :] * h_s_pd, a_w)
# Local stability
local_stb = da_dv
# String stability
string_stb = da_dv ** 2 - 2 * da_ds + 2 * da_dv * da_dvd

# Calculate equilibrium states
for equi_speed in range(5 * step, 30 * step + 1):
    subject_ind = np.where(all_input[:, 1] == equi_speed / step)
    subject_error = np.abs(acce_pred[subject_ind])
    equi_ind = np.argmin(subject_error)
    print('error is ', subject_error[equi_ind][0], 'and max error is ', np.max(subject_error), ' for speed ',
          equi_speed / step)
    subject_input = all_input[subject_ind]
    equi_gap = subject_input[equi_ind, 0]
    subject_local_stb, subject_string_stb = local_stb[subject_ind], string_stb[subject_ind]
    final_local_stb, final_string_stb = subject_local_stb[equi_ind], subject_string_stb[equi_ind]
    equi_state.append([equi_speed / step, equi_gap, final_local_stb[0], final_string_stb[0]])

# Plot the equilibrium states
equi_state = np.array(equi_state)
np.savetxt("ZMX_MLP_stb_theory.csv", equi_state, delimiter=',')
# m = np.polyfit(equi_state[:, 1], equi_state[:, 0], 1)
# print(m)
valid_string_stb = equi_state[:, -1]
color_string_stb = np.empty_like(valid_string_stb, dtype=str)
color_string_stb[np.where(valid_string_stb < 0)] = 'r'
color_string_stb[np.where(valid_string_stb > 0)] = 'b'
plt.scatter(equi_state[:, 1], equi_state[:, 0], c=color_string_stb)
# plt.plot(equi_state[:, 1], m[0] * equi_state[:, 1] + m[1], color='y')
plt.show()
