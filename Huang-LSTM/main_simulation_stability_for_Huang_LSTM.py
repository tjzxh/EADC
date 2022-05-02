import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import joblib
import tensorflow as tf

# Load Scaler
input_scaler = joblib.load("input_scaler.save")
output_scaler = joblib.load("output_scaler.save")


@tf.function
def model_predict(input_of_model, loaded_model):
    return loaded_model(input_of_model, training=False)


# Initialize
step, acce_bound = 0.1, 2
all_equi_speed = np.linspace(5.5, 20.5, 16)
all_equi_speed = all_equi_speed.tolist()
# full spacing:
# 5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20
# 3.438	5.445	7.689	9.449	12.122	14.754	17.048	20.782	25.678	31.396	39.082	47.998	58.115	69.94	87.985	108.745
all_equi_gap = [4.425, 6.616, 8.585, 10.824, 13.654, 16.064, 18.747, 23.096, 28.328, 35.023, 43.371, 52.927, 63.698,
                77.458, 117.841, 108.684]
for ind, equi_speed in enumerate(all_equi_speed):
    equi_gap = all_equi_gap[ind]
    # Create platoon array
    total_time, abnormal_step = int(10001), 0
    disturb_time = 600 if equi_speed < 19.1 else total_time
    position, speed = np.zeros((total_time - 1, 101)), np.zeros((total_time, 101))
    origin_input = np.zeros((total_time - 1, 101, 50, 3))
    abnormal_flag = np.zeros((total_time, 101))
    # Load the trained NN model
    model_name = "Huang_LSTM_0310"
    model = load_model(model_name)
    for veh_num in range(position.shape[1]):
        position[0][veh_num] = equi_gap * (100 - veh_num)
        speed[0][veh_num] = equi_speed
        origin_input[0][veh_num] = np.array([equi_speed, equi_gap, 0] * 50).reshape(50, 3)
        model_input = input_scaler.transform(origin_input[0][veh_num])
        model_output = model_predict(model_input[np.newaxis], model)
        final_output = output_scaler.inverse_transform(model_output)
        speed[1][veh_num] = final_output[0][0] if veh_num != 0 else equi_speed

    # Update speed and position with kinematic equation
    for time_step in range(1, total_time - 1):
        for veh_num in range(position.shape[1]):
            position[time_step][veh_num] = position[time_step - 1][veh_num] + (
                    speed[time_step - 1][veh_num] + speed[time_step][veh_num]) * step / 2
            # Calculate velocity with Huang-LSTM model
            if veh_num == 0:
                if time_step <= disturb_time:
                    speed[time_step + 1][veh_num] = equi_speed
                elif disturb_time < time_step <= disturb_time + 50:
                    speed[time_step + 1][veh_num] = speed[time_step][veh_num] - 0.5 * step
                elif disturb_time + 50 < time_step <= disturb_time + 100:
                    speed[time_step + 1][veh_num] = speed[time_step][veh_num] + 0.5 * step
                else:
                    speed[time_step + 1][veh_num] = equi_speed
            else:
                spacing = position[time_step][veh_num - 1] - position[time_step][veh_num]
                vd = speed[time_step][veh_num] - speed[time_step][veh_num - 1]
                # Purely use Huang-LSTM
                new_input = np.array([speed[time_step][veh_num], spacing, vd])
                origin_input[time_step][veh_num] = np.vstack((origin_input[time_step - 1][veh_num][1:], new_input))
                model_input = input_scaler.transform(origin_input[time_step][veh_num])
                model_output = model_predict(model_input[np.newaxis], model)
                final_output = output_scaler.inverse_transform(model_output)
                speed[time_step + 1][veh_num] = final_output[0][0]
                if speed[time_step + 1][veh_num] < 0:
                    speed[time_step + 1][veh_num] = 0
                    abnormal_step += 1

    print(abnormal_step, total_time, abnormal_step / total_time / 100)

    np.savetxt("Huang_LSTM_speed_simulation_" + str(equi_speed) + ".csv", speed, delimiter=',')
    np.savetxt("Huang_LSTM_position_simulation_" + str(equi_speed) + ".csv", position, delimiter=',')
    acce = np.diff(speed, axis=0) / step
    np.savetxt("Huang_LSTM_acce_simulation_" + str(equi_speed) + ".csv", acce, delimiter=',')

    # Plot speed deviation, acce and spacing
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(19.20, 9.83))
    for ind, acc in enumerate(acce.T):
        axs[0].plot(range(acce.shape[0]), acc, label="veh " + str(ind), lw=1)
    for ind, veh in enumerate(speed.T):
        axs[1].plot(range(total_time), veh, label="veh " + str(ind), lw=1)
    all_spacing = np.abs(np.diff(position))
    for ind, veh in enumerate(all_spacing.T):
        axs[2].plot(range(total_time - 1), veh, label="veh " + str(ind), lw=1)
    plt.savefig('Huang_LSTM_Simulation_' + str(equi_speed) + '.png', dpi=500, bbox_inches='tight')
