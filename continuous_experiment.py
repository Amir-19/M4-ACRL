from utils import *
from lib_robotis_hack import *
from dynamic_plotter import *
import numpy as np
from actor_critic_continuous import ActorCriticContinuous

import signal
import time
import pickle

cd1 = None
cd2 = None
flag_stop = False
plotting = False

def main():
    global cd1,cd2, flag_stop, plotting
    cd1 = []
    cd2 = []
    # servo connection step
    D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QD8V", baudrate=1000000)
    s1 = Robotis_Servo(D, 2)
    s2 = Robotis_Servo(D, 3)
    s1.move_angle(1.0,blocking=True)
    dir = 1
    i = 0

    if plotting:
        d1 = DynamicPlot(window_x = 150, title = 'Servo Controller state-reward-action', xlabel = 'time_step', ylabel= 'value')
        d1.add_line('Angular Position')
        d1.add_line('Reward * 2')
        d1.add_line('Selected Action')

        d2 = DynamicPlot(window_x = 150, title = 'Servo Controller mean and variance', xlabel = 'time_step', ylabel= 'value')
        d2.add_line('Mean')
        d2.add_line('Variance')


    n_bin = last_bin
    num_state = n_bin + 1
    actions = [-0.1,0.1]
    bins = np.linspace(0, 6, n_bin, endpoint=False)
    ac = ActorCriticContinuous(num_state)

    last_state = None
    last_action = None
    last_reward = None
    last_angle = None

    while True:
        # real time cumulant update
        # reading data for servo 1
        [ang, position, speed, load, voltage, temperature,direction] = read_data(s1)

        current_state = get_angle_bin(ang,direction,bins)
        if last_state is not None:
            mean, var = ac.update(feature_vector(last_state), last_reward, feature_vector(current_state), 0.4)
            # print "ang: ",ang," reward: ",last_reward
        action = ac.pick_action_from_dist(feature_vector(current_state))
        reward = reward_disc(ang)
        ang_to_go = ang + action / 10
        if ang_to_go > 1.5:
            ang_to_go = 1.5
        elif ang_to_go < -1.5:
            ang_to_go = -1.5
        s1.move_angle(ang_to_go,blocking=True)

        if plotting and last_state is not None:
            d1.update(i, [last_angle, last_reward*2,last_action])
            d2.update(i, [mean,var])
            cd1.append([i, last_angle, last_reward*2, last_action])
            cd2.append([i,mean,var])
        i += 1

        last_state = current_state
        last_angle = ang
        last_action = action / 10
        last_reward = reward

# write plotting data to file before ending by ctrl+c
def signal_handler(signal, frame):
    global flag_stop, cd1,cd2

    # stop threads
    flag_stop = True

    # now we need to dump the sensorimotor datastream to disk
    np_cd1 = np.asarray(cd1)
    np_cd2 = np.asarray(cd2)
    np.savetxt('continuous_exp_data_plot_sra.txt', np_cd1)
    np.savetxt('continuous_exp_data_plot_prob.txt', np_cd2)
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()







