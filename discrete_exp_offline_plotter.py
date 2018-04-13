import numpy
from dynamic_plotter import *
import time


def main():

    d1 = DynamicPlot(window_x = 150, title = 'Servo Controller state-reward-action', xlabel = 'time_step', ylabel= 'value')
    d1.add_line('Angular Position')
    d1.add_line('Reward * 2')
    d1.add_line('Chosen Action (0 for -0.1, 1 for +0.1)')

    d2 = DynamicPlot(window_x = 150, title = 'Servo Controller action probabilities', xlabel = 'time_step', ylabel= 'probability')
    d2.add_line('Action -0.1 Probability')
    d2.add_line('Action +0.1 Probability')


    # you need to choose the file you want to do the offline plot from
    s1_ds = numpy.loadtxt('discrete_exp_data_plot_sra.txt')
    s2_ds = numpy.loadtxt('discrete_exp_data_plot_prob.txt')
    for i in range(s1_ds.shape[0]):
        d1.update(i, [s1_ds[i][1],s1_ds[i][2],s1_ds[i][3]])
        d2.update(i,[s2_ds[i][1],s2_ds[i][2]])
        # change to make fast and slow plotting
        time.sleep(0.0)
    while True:
        pass
if __name__ == '__main__':
    main()