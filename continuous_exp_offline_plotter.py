import numpy
from dynamic_plotter import *
import time


def main():

    d1 = DynamicPlot(window_x = 150, title = 'Servo Controller state-reward-action', xlabel = 'time_step', ylabel= 'value')
    d1.add_line('Angular Position')
    d1.add_line('Reward * 2')
    d1.add_line('Selected Action')

    d2 = DynamicPlot(window_x = 150, title = 'Servo Controller mean and variance', xlabel = 'time_step', ylabel= 'value')
    d2.add_line('Mean')
    d2.add_line('Variance')


    # you need to choose the file you want to do the offline plot from
    s1_ds = numpy.loadtxt('continuous_exp_data_plot_sra.txt')
    s2_ds = numpy.loadtxt('continuous_exp_data_plot_prob.txt')
    for i in range(2800,s1_ds.shape[0]):
        d1.update(i, [s1_ds[i][1],s1_ds[i][2],s1_ds[i][3]])
        d2.update(i,[s2_ds[i][1],s2_ds[i][2]])
        # change to make fast and slow plotting
        time.sleep(0.0)
    while True:
        pass
if __name__ == '__main__':
    main()