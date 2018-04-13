from utils import *
from lib_robotis_hack import *
from dynamic_plotter import *
import numpy as np
from actor_critic_discrete import ActorCriticDiscrete


import signal
import time
import pickle

dd1 = None
dd2 = None
flag_stop = False
plotting = True

def main():
    global dd1,dd2, flag_stop, plotting
    dd1 = []
    dd2 = []

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
        d1.add_line('Chosen Action (0 for -0.1, 1 for +0.1)')

        d2 = DynamicPlot(window_x = 150, title = 'Servo Controller action probabilities', xlabel = 'time_step', ylabel= 'probability')
        d2.add_line('Action -0.1 Probability')
        d2.add_line('Action +0.1 Probability')

    n_bin = last_bin
    num_state = n_bin + 1
    actions = [-0.1,0.1]
    if FA_scheme == 1:
        bins = np.linspace(0, 6, n_bin, endpoint=False)
    else:
        bins = np.linspace(0, 3, n_bin, endpoint=False)
    ac = ActorCriticDiscrete(num_state,len(actions),0.1,0.1,0.01)

    last_state = None
    last_action = None
    last_reward = None
    last_angle = 0.0
    while True:
        # real time cumulant update
        # reading data for servo 1
        [ang, position, speed, load, voltage, temperature,direction] = read_data(s1)

        current_state = get_angle_bin(ang,direction,bins)
        if last_state is not None:
            ac.update(feature_vector(last_state), last_reward, feature_vector(current_state), 0.4)
            #print "ang: ",ang," reward: ",last_reward
        action, prob_actions = ac.pick_action_from_dist(feature_vector(current_state))
        reward = reward_disc(ang)

        ang_to_go = ang + actions[action]
        if ang_to_go > 1.5:
            ang_to_go = 1.5
        elif ang_to_go < -1.5:
            ang_to_go = -1.5
        s1.move_angle(ang_to_go,blocking=True)


        if plotting and last_state is not None:
            d1.update(i, [last_angle, last_reward*2, last_action])
            d2.update(i,[prob_actions[0],prob_actions[1]])
            dd1.append([i, last_angle, last_reward*2, last_action])
            dd2.append([i,prob_actions[0],prob_actions[1]])
        i += 1

        last_state = current_state
        last_angle = ang
        last_action = action
        last_reward = reward

# write plotting data to file before ending by ctrl+c
def signal_handler(signal, frame):
    global flag_stop, dd1,dd2

    # stop threads
    flag_stop = True

    # now we need to dump the sensorimotor datastream to disk
    np_dd1 = np.asarray(dd1)
    np_dd2 = np.asarray(dd2)
    np.savetxt('discrete_exp_data_plot_sra.txt', np_dd1)
    np.savetxt('discrete_exp_data_plot_prob.txt', np_dd2)
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()







