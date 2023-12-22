# -*- coding: utf-8 -*-
# 6.100B Fall 2023
# Problem Set 4: Sea Level Rise
# Name: Justin Chen
# Collaborators:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.stats as st
from scipy.interpolate import interp1d

#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 97.5th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 97.5th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper - mean) / st.norm.ppf(.975)

def load_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())


###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100, inclusive, and not included in the data, 
    the values for that year should be interpolated. If show_plot, displays a 
    plot with mean and the 95% confidence interval, assuming sea level rise 
    follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing the year, the mean, the 2.5th 
        percentile, 97.5th percentile, and standard deviation of the sea level rise
        for the years between 2020-2100 inclusive
	"""
    lines = ([[int(x.split(',')[0]), *map(float, x.split(',')[1:])] for x in open("sea_level_change.csv", 'r').read().split('\n')[1:]])
    years = np.array([])
    slows = np.array([])
    fasts = np.array([])
    means = np.array([])
    sds = np.array([])
    for i in range(len(lines)-1):
        first, second = lines[i],lines[i+1]
        first_year, second_year = first[0], second[0]
        these_years = np.linspace(first_year, second_year, 11)

        first_slow_slr, second_slow_slr = first[1], second[1]
        these_slows = np.linspace(first_slow_slr, second_slow_slr, 11)

        first_fast_slr, second_fast_slr = first[2], second[2]
        these_fasts = np.linspace(first_fast_slr, second_fast_slr, 11)

        these_means = (these_slows+these_fasts)/2
        these_sds = (these_fasts-these_slows)/(2*st.norm.ppf(0.975))
        years = np.append(years, these_years[0:10])
        slows = np.append(slows, these_slows[0:10])
        fasts = np.append(fasts, these_fasts[0:10])
        means = np.append(means, these_means[0:10])
        sds = np.append(sds, these_sds[0:10])
    last_line = lines[-1]
    years = np.append(years, last_line[0])
    slows = np.append(slows, last_line[1])
    fasts = np.append(fasts, last_line[2])
    means = np.append(means, (last_line[1]+last_line[2])/2)
    sds = np.append(sds, (last_line[2]-last_line[1])/(2*st.norm.ppf(0.975)))

    final_array = np.vstack((years,means, slows, fasts, sds))
    final_array = np.transpose(final_array)
    if show_plot:
        plt.plot(final_array[:, 0], final_array[:,1])
        plt.plot(final_array[:, 0], final_array[:,2], linestyle = 'dashed')
        plt.plot(final_array[:, 0], final_array[:,3], linestyle = 'dashed')

        plt.show()
    #final_array = np.vstack((final_array, np.array([last_line[0], (last_line[1]+last_line[2])/2, last_line[1], last_line[2], (last_line[2]-last_line[1])/4])))
    return final_array
        



def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution. 

    (This function should use the predicted_sea_level_rise function.)

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    index = np.where(data[:, 0] == year)
    mean = data[index,1]
    sd = data[index,4]
    samples = []
    for _ in range(num):
        samples.append((np.random.normal(mean, sd))[0,0])
    return np.array(samples)

    
def plot_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    all_samples = []
    for year in data[:, 0]:
        samples = simulate_year(data, year, 500)
        all_samples.append(samples)
    plt.plot(data[:, 0], data[:,1])
    plt.plot(data[:, 0], data[:,2], linestyle = 'dashed')
    plt.plot(data[:, 0], data[:,3], linestyle = 'dashed')
    for i in range(len(data[:, 0])):
        plt.scatter(np.full_like(all_samples[i], data[i, 0]), all_samples[i], s=4, alpha = 0.2, c='gray')
    plt.show()
    return

##########
# Part 2 #
##########

def simulate_water_levels(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a python list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    simulated_levels = []
    for i in range(np.size(data[:, 0])):
        year = data[i, 0]
        estimate = simulate_year(data, year, 1)[0]
        simulated_levels.append(estimate)
    return simulated_levels


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=500000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		a python list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    slrs = water_level_loss_no_prevention[:, 0]
    percentages = water_level_loss_no_prevention[:, 1]
    interpolation_func = interp1d(slrs, percentages, kind = 'linear', fill_value = 'extrapolate')
    costs = house_value/1000*interpolation_func(water_level_list)/100
    costs = list(costs)
    costs = [0 if x<0 else x for x in costs]
    costs = [house_value/1000 if x>house_value/1000 else x for x in costs]
    return costs

def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=500000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having two consecutive years with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have two consecutive years with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    repair_only_costs = repair_only(water_level_list, water_level_loss_no_prevention, house_value)
    stop_at = len(repair_only_costs)-1
    
    for i in range(len(repair_only_costs)-1):
        if (repair_only_costs[i] > cost_threshold/1000) and (repair_only_costs[i+1] > cost_threshold/1000):
            stop_at = i+1
            break

    slrs = water_level_loss_no_prevention[:, 0]
    percentages_before = water_level_loss_no_prevention[:, 1]
    percentages_after = water_level_loss_with_prevention[:, 1]
    interpolation_func_before = interp1d(slrs, percentages_before, kind = 'linear', fill_value = 'extrapolate')
    interpolation_func_after = interp1d(slrs, percentages_after, kind='linear', fill_value='extrapolate')
    costs_before = list(house_value/1000*interpolation_func_before(water_level_list[0:stop_at+1])/100)
    costs_after = list(house_value/1000*interpolation_func_after(water_level_list[stop_at+1:])/100)
    costs = costs_before + costs_after
    costs = [0 if x<0 else x for x in costs]
    costs = [house_value/1000 if x>house_value/1000 else x for x in costs]
    
    return costs



def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=500000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    return repair_only(water_level_list, water_level_loss_with_prevention, house_value)


def plot_prep_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=500000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 2.5th percentile, 97.5th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    plt.plot()
    repair_only_total = np.zeros(81)
    wait_total = np.zeros(81)
    prepare_total = np.zeros(81)
    for i in range(500):
        water_levels = simulate_water_levels(data)
        repair_only_costs = np.array(repair_only(water_levels, water_level_loss_no_prevention, house_value))
        wait_costs = np.array(wait_a_bit(water_levels, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold))
        prepare_immediately_costs = np.array(prepare_immediately(water_levels, water_level_loss_with_prevention, house_value))
        plt.scatter(data[:, 0], repair_only_costs, c = 'red', s = 3, alpha = 0.3)
        plt.scatter(data[:, 0], wait_costs, c = 'yellow', s = 3, alpha = 0.3)
        plt.scatter(data[:, 0], prepare_immediately_costs, c = 'blue', s=3, alpha = 0.3)
        repair_only_total += repair_only_costs
        wait_total += wait_costs
        prepare_total += prepare_immediately_costs
    repair_average = repair_only_total/500
    wait_average = wait_total/500
    prepare_average = prepare_total/500
    plt.plot(data[:, 0], repair_average, c = 'red', label = 'repair-only scenario')
    plt.plot(data[:, 0], wait_average, c = 'yellow', label = 'wait a-bit scenario')
    plt.plot(data[:, 0], prepare_average, c = 'blue', label = 'prepare-immediately scenario')
    plt.legend()
    plt.show()
    return





if __name__ == '__main__':
    # Comment out the 'pass' statement below to run the lines below it
    

    # Uncomment the following lines to plot generate plots
    data = predicted_sea_level_rise(show_plot=True)
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    plot_simulation(data)
    plot_prep_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)