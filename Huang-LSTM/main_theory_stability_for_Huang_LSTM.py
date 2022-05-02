import tensorflow as tf
from keras.models import load_model
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Load Scaler
input_scaler = joblib.load("input_scaler.save")
output_scaler = joblib.load("output_scaler.save")
v_std, s_std, vd_std = input_scaler.scale_[0], input_scaler.scale_[1], input_scaler.scale_[0]
# Load the trained model
model_name = "Huang_LSTM_0310"
model = load_model(model_name)
# Construct input data
equi_state = []
step, speed_list = 1000, [20]#np.linspace(20, 30, 11)  # list(range(5, 21))
# gap = [9.449, 12.122, 14.754]
# headway_list = [0.9, 1.175, 1.48, 1.73333333, 2.24285714, 3., 3.88333333]
# full headway: [0.9        1.175      1.48       1.73333333 2.24285714 3.       3.88333333]
for ind, equi_speed in enumerate(speed_list):
    all_input, flat_input = [], []
    # step = 1000 if equi_speed < 11 else 100
    for possible_gap in range(int(90 * step), int(110 * step)):
        single_input = [equi_speed, possible_gap / step, 0]
        flat_input.append(single_input)
        all_input += single_input * 50
        # all_input = single_array if not all_input.shape[0] else np.vstack((all_input, single_array))
    print('Finish creating input data')
    all_input = np.array(all_input).reshape(len(flat_input) * 50, 3)
    input_scaled = input_scaler.transform(all_input)
    input_scaled = input_scaled.flatten()
    all_input = input_scaled.reshape((len(flat_input), 50, 3))
    all_input = all_input.astype("float32")

    # Calculate partial differiential for stability analysis
    T_array = np.linspace(5, 0.1, 50).reshape((50, 1))
    T_squre_array = -pow(np.linspace(5, 0.1, 50).reshape((50, 1)), 2)
    x_test4grad = tf.convert_to_tensor(all_input)
    x_grad = tf.Variable(x_test4grad)
    with tf.GradientTape() as tape:
        y_pred = model(x_grad, training=False)
    grads = tape.gradient(y_pred, x_grad)
    grads = grads.numpy()
    f_s, f_v, f_vd = grads[:, :, 1] * v_std / s_std, grads[:, :, 0], grads[:, :, 2] * v_std / vd_std

    dsize = f_v.shape[0]
    f_s_array, f_v_array, f_vd_array = np.sum(f_s, axis=1), np.sum(f_v, axis=1), np.sum(f_vd, axis=1)
    f_s_array, f_v_array, f_vd_array = f_s_array.reshape((dsize, 1)), f_v_array.reshape((dsize, 1)), f_vd_array.reshape(
        (dsize, 1))
    # Loacl stability
    local_stb1 = -(1 - (f_v_array + f_vd_array) - np.dot(f_s, T_array)) / np.dot(f_v + f_vd, T_array)
    local_stb2 = -f_s_array / np.dot(f_v + f_vd, T_array)
    # String stability
    string_stb = f_v_array * np.dot(f_v, T_squre_array) + np.dot(f_v, T_squre_array) * f_vd_array + f_v_array * np.dot(
        f_vd, T_squre_array) - np.dot(f_v + f_vd, T_squre_array) + np.dot(f_v, T_squre_array) * np.dot(f_s,
                                                                                                       T_array) - np.dot(
        f_s, T_squre_array) * np.dot(f_v, T_array)
    necessary_string_stb = 1 + pow(f_v_array, 2) + 2 * f_v_array * f_vd_array - 2 * (
            f_v_array + f_vd_array) - 2 * np.dot(f_s, T_array) + 2 * f_v_array * np.dot(f_s, T_array) - 2 * np.dot(
        f_v, T_array) * f_s_array

    # (np.dot(f_s, T_array) - np.sum(f_v, axis=1).reshape((f_v.shape[0], 1)) * np.dot(f_s, T_array))
    prediction = output_scaler.inverse_transform(y_pred)
    # print(prediction, local_stb, string_stb)
    speed_pred = prediction

    # Calculate equilibrium states
    flat_input = np.array(flat_input)
    subject_error = np.abs(speed_pred - equi_speed)
    equi_ind = np.argmin(subject_error)
    print('error is ', subject_error[equi_ind][0], 'and max error is ', np.max(subject_error), ' for speed ',
          equi_speed)
    equi_gap = flat_input[equi_ind, 1]
    final_local_stb1, final_local_stb2, final_string_stb, final_necessary_stb = local_stb1[equi_ind], local_stb2[
        equi_ind], string_stb[
                                                                                    equi_ind], necessary_string_stb[
                                                                                    equi_ind]
    equi_state.append(
        [equi_speed, equi_gap, final_local_stb1, final_local_stb2, final_string_stb[0], final_necessary_stb[0]])

# Plot the equilibrium states
equi_state = np.array(equi_state)
print(equi_state[:, 0], equi_state[:, 1], equi_state[:, 1] / equi_state[:, 0], equi_state[:, -1])
np.savetxt("stb_theory" + str(speed_list[0]) + ".csv", equi_state, delimiter=',')
# m = np.polyfit(equi_state[:, 1], equi_state[:, 0], 1)
# print(m)
plt.figure(1)
valid_string_stb1, valid_string_stb2 = equi_state[:, 4], equi_state[:, 5]
color_string_stb = np.empty_like(valid_string_stb1, dtype=str)
color_string_stb[np.where(np.logical_or(valid_string_stb1 < 0, valid_string_stb2 < 0))] = 'r'
color_string_stb[np.where(np.logical_and(valid_string_stb1 > 0, valid_string_stb2 > 0))] = 'b'
plt.scatter(equi_state[:, 1], equi_state[:, 0], c=color_string_stb)
plt.title('string stability', fontsize=16)
# plt.plot(equi_state[:, 1], m[0] * equi_state[:, 1] + m[1], color='y')

# plot local stability
plt.figure(2)
valid_local_stb1, valid_local_stb2 = equi_state[:, 2], equi_state[:, 3]
color_local_stb = np.empty_like(valid_local_stb1, dtype=str)
color_local_stb[np.where(np.logical_or(valid_local_stb1 > 0, valid_local_stb2 > 0))] = 'r'
color_local_stb[np.where(np.logical_and(valid_local_stb1 < 0, valid_local_stb2 < 0))] = 'b'
plt.scatter(equi_state[:, 1], equi_state[:, 0], c=color_local_stb)
plt.title('local stability', fontsize=16)
plt.show()
