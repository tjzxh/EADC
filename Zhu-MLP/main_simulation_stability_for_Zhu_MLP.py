import matplotlib.pyplot as plt
import numpy as np
from simulation_env import Env
from ZMX_RL_MLP import DDPG

TTC_threshold = 4
env = Env(TTC_threshold)
s_dim = env.n_features
a_dim = env.n_actions
a_bound = env.action_Bound

# Create platoon array
total_time, abnormal_step = int(10001), 0
position, speed, acce = np.zeros((total_time - 1, 101)), np.zeros((total_time - 1, 101)), np.zeros((total_time, 101))
abnormal_flag = np.zeros((total_time, 101))
# Load the trained NN model
base_name = f'origin_4.001_NOCA'
model_file = f'save/model_{base_name}'
ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg.restore(model_file)
# Initialize
step = 0.1
all_speed = np.linspace(5, 30, 51)
all_gap = [element * 3.22 for element in all_speed]
for equi_speed, equi_gap in zip(all_speed, all_gap):
    # Simulation start
    for veh_num in range(position.shape[1]):
        position[0][veh_num] = equi_gap * (100 - veh_num)
        speed[0][veh_num] = equi_speed
        acce[0][veh_num] = 0
        model_input = np.array([equi_gap, equi_speed, 0]).reshape(1, 3)
        acce[1][veh_num] = ddpg.choose_action(model_input)[0] if veh_num != 0 else 0

    # Update speed and position with kinematic equation
    for time_step in range(1, total_time - 1):
        for veh_num in range(position.shape[1]):
            speed[time_step][veh_num] = speed[time_step - 1][veh_num] + acce[time_step - 1][veh_num] * step
            if speed[time_step][veh_num] < 0:
                speed[time_step][veh_num] = 0
            position[time_step][veh_num] = position[time_step - 1][veh_num] + max(
                speed[time_step - 1][veh_num] * step + acce[time_step - 1][veh_num] * pow(step, 2) / 2, 0)
            # Calculate acceleration with NN CF model
            if veh_num == 0:
                if time_step <= 600:
                    acce[time_step][veh_num] = 0
                elif 600 < time_step <= 650:
                    acce[time_step][veh_num] = -0.5
                elif 650 < time_step <= 700:
                    acce[time_step][veh_num] = 0.5
                else:
                    acce[time_step][veh_num] = 0
            else:
                spacing = position[time_step][veh_num - 1] - position[time_step][veh_num]
                vd = speed[time_step][veh_num] - speed[time_step][veh_num - 1]
                # safety trick for NN CF model
                if spacing <= 1 * speed[time_step][veh_num]:
                    acce[time_step][veh_num] = -min(a_bound, abs(
                        pow(speed[time_step][veh_num], 2) - pow(speed[time_step][veh_num - 1], 2)) / 2 / spacing)
                    abnormal_step += 1
                    # print(spacing, speed[time_step][veh_num], -vd)
                else:
                    model_input = np.array([spacing, speed[time_step][veh_num], vd]).reshape(1, 3)
                    acce[time_step][veh_num] = ddpg.choose_action(model_input)[0]
                    # print(spacing, speed[time_step][veh_num], -vd, acce[time_step + 1][veh_num])

    print(abnormal_step, total_time, abnormal_step / total_time / 100)

    np.savetxt("ZMX_MLP_" + str(equi_speed) + "_speed_simulation.csv", speed, delimiter=',')
    np.savetxt("ZMX_MLP_" + str(equi_speed) + "_position_simulation.csv", position, delimiter=',')
    np.savetxt("ZMX_MLP_" + str(equi_speed) + "_acce_simulation.csv", acce, delimiter=',')
    #
    # Plot speed deviation, acce and spacing
    fig, axs = plt.subplots(1, 3, sharex=True)
    for ind, acc in enumerate(acce.T):
        axs[0].plot(range(total_time), acc, label="veh " + str(ind), lw=1)
    # axs[0].plot(range(total_time), [0] * total_time, label="equilibrium acce", ls='--', lw=1)
    for ind, veh in enumerate(speed.T):
        axs[1].plot(range(total_time - 1), veh, label="veh " + str(ind), lw=1)
    # axs[1].plot(range(total_time - 1), [equi_speed] * (total_time - 1), label="equilibrium speed", ls='--', lw=1)
    all_spacing = np.abs(np.diff(position))
    # all_spacing = all_spacing[:, 1:]
    for ind, veh in enumerate(all_spacing.T):
        axs[2].plot(range(total_time - 1), veh, label="veh " + str(ind), lw=1)
    # axs[2].plot(range(total_time - 1), [equi_gap] * (total_time - 1), label="equilibrium gap", ls='--', lw=1)
    # plt.legend()
    axs[0].set_ylabel('acce (m/s/s)')
    axs[1].set_ylabel('speed (m/s)')
    axs[2].set_ylabel('spacing (m)')
    plt.title("ZMX_MLP_" + str(equi_speed), fontsize=16)
    plt.savefig("ZMX_MLP_" + str(equi_speed) + ".png", dpi=800, bbox_inches='tight')
    # plt.show()
