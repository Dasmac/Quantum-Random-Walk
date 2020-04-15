from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
from scipy import optimize
from random import randint
from matplotlib import cm
from multiprocessing import Pool
import time

#  Run program for 30 < R < 80
#  This Program is for the single node only - along the x axis - therefore the condition for the 1 arrest rate is that
#  to check if y == 0 - self.cartesian_positions is a 2d array of form [x][y]
#  Create a heatmap from a cont map to display probability diagram --- DONE
#  Plot Time taken and accuracy against R and Total walks ---------------- DO
#  Create maps for other energies and confirm results to through -->>>
#  https://www.st-andrews.ac.uk/physics/quvis/simulations_html5/sims/2DCircularWell/infinite%20circular%20well5.html
class DataStruct:
    def __init__(self, fit_xdata, fit_ydata, R, num_walks, runtime):
        self.x_data = fit_xdata
        self.y_data = fit_ydata
        self.R = R
        self.num_walks = num_walks
        self.runtime = runtime


class Particle:
    def __init__(self, J):
        self.J = J
        self.alive = True
        self.steps = 0
        self.dimensions = 1
        self.x_coord = 0

    def chance_Execution(self):
        # arrest would normally be a function of the position of the particle
        for dim in range(self.dimensions):
            if np.abs(self.x_coord) == np.abs(self.J):
                self.alive = False

    def walk_Particle(self):
        while self.alive is True:
            if rnd.random() < 0.5:
                self.x_coord += 1
            else:
                self.x_coord -= 1

            self.chance_Execution()
            if self.alive is True:
                self.steps += 1


def Sort_Dict_Key_Order(endpoint_tep_lengths):  # Sorting the dictionary into key order
    sorted_endpoint_step_DICT = {}
    #  print(max(endpoint_step_lengths.keys()))
    for key in range(int(max(endpoint_step_lengths.keys())) + 1):
        sorted_endpoint_step_DICT[key] = endpoint_step_lengths.get(key, 0)  # creates a new kvp
        # if there isn't one already existing setting the value to 0
        # if exists then just assign the key in sorted dict a new value
    return sorted_endpoint_step_DICT


def func(x, a, c):
    return a*x + c


def simulate_i(args):
    # i is passed in from the pool.map(simulate_i, particles) ---- particles = [(0, particle stuff), (1, particle stuff)
    k, particle = args
    particle.walk_Particle()  # Begins the walk, including the handleing of the arrest
    '''
    distance_from_origin = 0
    for dim in range(p.dimensions):
        distance_from_origin += (p.cartesian_positions[dim]) ** 2
    '''
    # print(k, particle.steps)
    return particle


def get_max_turning_point(R, dict):
    # for even R values only the odd time steps survived will be non-zero
    max_tp_index = 0

    if R % 2 == 0:
        # first index=0 value is actually starting at 1, increments in 2's
        values = list(dict.values())[1::2]
        keys = list(dict.keys())[1::2]
        for index in range(int(R/2-1), len(values)):
            pre_peak = (values[index]-values[index-2])/3
            post_peak = (values[index+2]-values[index])/6
            if post_peak < 0 < pre_peak:
                max_tp_index = index
                break
    '''
    else:
        pass
   '''
    if max_tp_index == 0:
        print('Could not find maximum!!!')
        return R

    return max_tp_index


if __name__ == "__main__":
    Rs = [24, 30, 40, 60, 80]
    # Rs = [8, 12, 16, 20, 24]
    num_walks_set = [250, 2500, 25000, 250000, 2500000, 10000000]
    # num_walks_set = [25, 20, 20, 200, 2500, 1000]
    # data is a 3x3 array of objects for storing the data structs in
    data = np.empty(shape=(len(num_walks_set), len(Rs)), dtype=object)
    #  the use of rth and walkth is to keep track of the index's so we may change their object value later
    for rth, R in enumerate(Rs):
        for walkth, num_walks in enumerate(num_walks_set):
            print(f'Starting walk of R = {R} and {num_walks} walks')
            # min R = 8
            R = R
            num_walks = num_walks
            endpoint_step_lengths = {}
            walk_count_grid = np.zeros(((R * 2 + 1), (R * 2 + 1)), dtype=int)
            x_y_ranges = np.arange(-R, R, 1)

            start = time.time()
            particles = [(i, Particle(J=R)) for i in range(num_walks)]
            # Using 10 threads
            pool = Pool(10)
            # funciton to pass through the map bit is Particle.Walk_Particle
            particli = pool.map(simulate_i, particles)
            stop = time.time()
            runtime = stop - start

            # from here we only need to bin the steps and
            print('Runtime: ', (stop - start))
            for p in particli:
                steps_survived = p.steps
                endpoint_step_lengths[steps_survived] = endpoint_step_lengths.get(steps_survived, 0) + 1

            # Following lines clean up memory
            for q in particli:
                del p

            endpoint_step_lengths_sorted_DICT = Sort_Dict_Key_Order(endpoint_step_lengths)  # Sort the bins
            cum_Dist = np.cumsum(np.array(list(endpoint_step_lengths_sorted_DICT.values()))[::-1])[::-1]
            conv_from_Dict_to_Arr = np.array(list(endpoint_step_lengths_sorted_DICT.keys()))

            # get max function obtains the start for the exponential curve
            start_step_survived = get_max_turning_point(R, endpoint_step_lengths_sorted_DICT)
            # start_step_survived = 11
            cutoff = -1
            fit_xdata = conv_from_Dict_to_Arr[start_step_survived::2]
            fit_ydata = np.log(cum_Dist[start_step_survived::2])
            popt, pcov = optimize.curve_fit(f=func, xdata=fit_xdata, ydata=fit_ydata)
            print(f'Lambda is {-popt[0]} & Variance is {pcov[0][0]}')
            print('grad, intercept --> ', popt)
            print('Covariance matrix: \n', pcov)

            data[walkth][rth] = DataStruct(fit_xdata, fit_ydata, R, num_walks, runtime)

    # Plot configuration
    # num_walks_set length must be divisible by 2!!!!!
    nrows = int(len(num_walks_set) / 2)
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    # data is a 3x3 array of objects
    fig.tight_layout(pad=2.0)
    colour_scheme = ['#27ee52', '#ad1dee', '#1d2bee', '#1deede', '#247442', '#f21d1d']
    for walk, walks in enumerate(data):
        ax = plt.subplot(nrows, ncols, walk+1)
        print(f'Walk is {walk}')
        for rth, R in enumerate(walks):
            # Building the plots
            # Chopping the data
            # Use of a third doesnt seem to work below - this may be due to a rounding issue under the hood
            chopped_x = R.x_data[:int(np.round((R.x_data[-1] * 0.6666)))]
            chopped_y = R.y_data[:int(np.round((R.x_data[-1] * 0.6666)))]
            popt, pcov = optimize.curve_fit(f=func, xdata=chopped_x, ydata=chopped_y, p0=[0.005, np.log(R.num_walks)])
            # The error in the intercept can be obtained from the pcov matrix
            y_results = popt[0] * R.x_data + popt[1]

            ax.plot(R.x_data, y_results, linewidth=1)
            ax.plot(R.x_data, R.y_data, label=f'J = {R.R}', color=colour_scheme[rth])
            ax.axvline(x=(R.x_data[-1] * 2/3), color=colour_scheme[rth])
            walks_const = R.num_walks

            # Building the data files
            simulation_parameters = pd.DataFrame(data={'R': [R.R], 'Walks': [R.num_walks]})
            time_steps_survived_frame = pd.DataFrame({'keys': list(R.x_data),
                                                      'values': list(R.y_data)})
            chopped_time_steps_survived_frame = pd.DataFrame({'keys': list(chopped_x),
                                                              'values': list(chopped_y)})

            sim_state = f'{R.R}_{R.num_walks}_{R.runtime}'  # For some reason the below doesnt like '' in the argument???
            time_steps_survived_frame.to_csv(
                f'C:\\Users\\willm\\OneDrive\\Desktop\\Comp labs\\Cirular potential programs\\Final programs\\Square well\\steps_survied_data_{sim_state}.csv',
                index=False)
            simulation_parameters.to_csv(
                f'C:\\Users\\willm\\OneDrive\\Desktop\\Comp labs\\Cirular potential programs\\Final programs\\Square well\\simulation_parameters_{sim_state}.csv',
                index=False)
            chopped_time_steps_survived_frame.to_csv(
                f'C:\\Users\\willm\\OneDrive\\Desktop\\Comp labs\\Cirular potential programs\\Final programs\\Square well\\simulation_parameters_{sim_state}.csv',
                index=False)

            print(rth)
        ax.legend(loc=1)
        ax.grid()
        ax.set_ylabel('Natural log of the number of walkers')
        ax.set_xlabel('Number of time steps')
        ax.set_title(f' {walks_const} walks completed.')

    # fig.suptitle('Natural Log of the ammount of walkers surviving N timesteps \n versus the Number of timesteps fontsize=20)
    plt.savefig('Log plots of varying walk count_for second half')
    plt.show()

    # Saving of the information
    # note that the .values and .keys must be converted into a list for any dictionary
    print()












