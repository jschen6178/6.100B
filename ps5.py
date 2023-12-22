# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Justin Chen
# Collaborators: N/A

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        # NOTE: TO BE IMPLEMENTED IN PART 4B.2 OF THE PSET
        yearly_city_temps = []
        for year in years:
            city_averages = []
            for city in cities:
                city_temps = []
                for x in open('data.csv', 'r').readlines():        
                        if str.upper(city) in x and int(x.split(',')[-1][:4]) == year:
                            city_temps.append(float(x.split(',')[1]))
                city_averages.append(sum(city_temps)/len(city_temps))
            yearly_city_temps.append(sum(city_averages)/len(city_averages))

        return yearly_city_temps
def linear_regression(x, y):
    x_diff = x-np.ones(np.size(x))*np.mean(x)
    y_diff = y-np.ones(np.size(y))*np.mean(y)
    m = np.dot(x_diff, y_diff)/np.dot(x_diff, x_diff)
    b = np.mean(y) - m*np.mean(x)
    return (m,b)

def squared_error(x, y, m, b):
    y_pred = m*x + np.ones(np.size(x))*b
    y_diff = y_pred - y
    tot_se = np.dot(y_diff, y_diff)
    return tot_se


def generate_polynomial_models(x, y, degrees):
    coeff_list = []
    for degree in degrees:
        coeff_list.append(np.polyfit(x, y, degree))
    return coeff_list


def evaluate_models(x, y, models, display_graphs=False):
    r2_values = []
    if display_graphs:
        figure = plt.figure()
    for i in range(len(models)):
        model = models[i]
        
        y_pred = np.polyval(model, x)
        r2 = r2_score(y, y_pred)
        r2_values.append(r2)
        if display_graphs:
            ax = figure.add_subplot(len(models), 1, i+1)
            ax.plot(x, y, color = 'red')
            ax.plot(x , y_pred, color = 'blue')
    if display_graphs:
        plt.show()
    return r2_values


def get_max_trend(x, y, length, positive_slope):
    i_return = 0
    j_return = length
    greatest_slope = 0
    #positive slope case
    if positive_slope:
        i=0
        while i+length <= np.size(x):
            j = i+length
            slope = linear_regression(x[i:j], y[i:j])[0]
            if slope > greatest_slope:
                greatest_slope = slope
                i_return = i
                j_return = i+length
            i += 1
    #negative slope case
    else:
        i=0
        while i+length <= np.size(x):
            j = i+length
            slope = linear_regression(x[i:j], y[i:j])[0]
            if slope < greatest_slope:
                greatest_slope = slope
                i_return = i
                j_return = i+length
            i += 1
    if greatest_slope == 0:
        return None
    else:
        return (i_return, j_return, greatest_slope)
        


def get_all_max_trends(x, y):
    
    if np.size(x) < 2:
        return []
    ret_list = []
    for length in range(2, np.size(x)+1):
        pos_trend = get_max_trend(x,y,length,True)
        neg_trend = get_max_trend(x,y,length,False)
        if pos_trend == None and neg_trend == None:
            ret_list.append((0, length, None))
        elif pos_trend == None:
            ret_list.append(neg_trend)
        elif neg_trend == None:
            ret_list.append(pos_trend)
        else:
            if abs(pos_trend[2]) > abs(neg_trend[2]):
                ret_list.append(pos_trend)
            else:
                ret_list.append(neg_trend)
    return ret_list

x = np.array(range(21))
y = (x-10)**2
print(get_max_trend(x,y,8,True))
result = get_all_max_trends(x,y)
print(result)
def calculate_rmse(y, estimated):
    raise NotImplementedError


def evaluate_rmse(x, y, models, display_graphs=False):
    raise NotImplementedError


if __name__ == '__main__':

    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    x = np.array(range(1961,2017))
    z = []
    for line in open("data.csv", 'r').readlines():
        
        if 'PHOENIX' in line:
            
            if line[-5:-1] == '1201':
                
                z.append(float(line.split(',')[1]))
                
    
    y = np.array(z)    
    coeffs_1d = generate_polynomial_models(x, y, [1,2,3])
    eval_1d = evaluate_models(x, y, coeffs_1d, display_graphs=True)

    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    my_dataset = Dataset('data.csv')
    phoenix_avgs = my_dataset.calculate_annual_temp_averages(['PHOENIX'], range(1961,2017))
    
    coeff = generate_polynomial_models(range(1961, 2017), phoenix_avgs, [1])
    r2 = evaluate_models(range(1961, 2017), phoenix_avgs, coeff, True)
    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    seattle_avgs = my_dataset.calculate_annual_temp_averages(['SEATTLE'], range(1961,2017))
    (pos_i, pos_j, slope) = get_max_trend(range(1961,2017), seattle_avgs, 30, True)
    print(pos_i)
    x_max = range(1961, 2017)[pos_i:pos_j]
    y_max = seattle_avgs[pos_i:pos_j]
    plt.plot(x_max, y_max)
    plt.show()
    ##################################################################################
    # Problem 5C: DECREASING TRENDS


    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.

    ##################################################################################
    # Problem 6B: PREDICTING


    ####################################################################################