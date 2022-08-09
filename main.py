import numpy as np
from policy import *
from problem_instance import *
import sys
from os.path import isfile
import os
import cProfile
from graphing_functions import *
from simulation import *
import matplotlib.pyplot as plt
import pickle
np.set_printoptions(threshold=sys.maxsize)


# These functions generate a list with a customer objects for each period in the selling horizon as elements
def single_customer_type(customer_list, seed):
    return [customer_list[0]]*T


def randomize_multiple_customer_types(customer_list, variance, T, seed):
    """
    Returns a list containing a set of integers, randomly distributed

    Parameters
    __________
    dimension: integer
        each element contains an integer between 0 and dimension-1
    variance: float
        the variance of the dirichlet distribution is given by variance*dimension, as described in this
        paper https://doi.org/10.1287/mnsc.2014.1939
    length: integer
        the length of the list

    Returns
    _______
    a list where each element contains an integer between 0 and dimension-1, and these integers are chosen
    randomly, where the amount each integer shows up in the list is given by a dirichlet distribution, and the elements
    are randomly permuted.
    """
    # Converts the variance into the required input parameter for the dirichlet distribution
    # that will achieve that variance
    num_customers = len(customer_list)
    a = (1 / num_customers) * ((num_customers - 1) / variance ** 2 - 1)
    alpha = (a,) * num_customers

    def return_function(customer_list, seed):
        np.random.seed(seed)
        c = np.random.dirichlet(alpha, 1)[0]
        # we have to scale the dirichlet distribution to the length of the list. Each entry in the dirichlet vector
        # is the number of times that the index of that entry shows up in the final list. We round the output, since
        # this must be an integer.
        c *= T
        c = np.round(c).astype(int)
        # as a result of the rounding, the parameters may no longer sum to length of the list, so we adjust
        while np.sum(c) > T:
            c[np.argmax(c)] -= 1
        while np.sum(c) < T:
            c[np.argmin(c)] += 1
        z = []
        y = 0
        for x in c:
            z += [y] * x
            y += 1
        # After putting the correct quantities of each integer in a list, we permute the integers
        z = np.random.permutation(z)
        arriving_customer_types = [customer_list[z[t]] for t in range(T)]
        return arriving_customer_types
    return return_function


# These functions save data as files, so trials which have a long computing time only need to be run once
def save_single_instance_data(revenue, inventory_vectors, offered_sets, cumulative_revenue, names, title):
    # save the data as recieved by the simulation
    pickle_data = revenue, inventory_vectors, offered_sets, cumulative_revenue, names

    directory = r"experiment_data/single_instances/"
    filename = directory + title + ".pickle"

    # We don't overwrite files, because we are not monsters!
    i = 0
    while isfile(filename):
        filename = directory + title + "_" + str(i) + ".pickle"
        i += 1

    # Save the data to the file
    with open(filename, 'wb') as handle:
        pickle.dump(pickle_data, handle)


def save_parameter_experiment_data(revenue_vs_parameter, policy_names, parameter_name, num_runs, title):
    pickle_data = revenue_vs_parameter, policy_names, parameter_name, num_runs

    directory = r"experiment_data/parameter_experiments/"
    filename = directory + title + ".pickle"

    # We don't overwrite files, because we are not monsters!
    i = 0
    while isfile(filename):
        filename = directory + title + "_" + str(i) + ".pickle"
        i += 1

    with open(filename, 'wb') as handle:
        pickle.dump(pickle_data, handle)


def open_single_instance_data(title):
    directory = r"experiment_data/single_instances/"
    filename = directory + title + ".pickle"
    with open(filename, 'rb') as f:
        output = pickle.load(f)
    return output


def open_parameter_experiment_data(title):
    directory = r"experiment_data/parameter_experiments/"
    filename = directory + title + ".pickle"
    with open(filename, 'rb') as f:
        output = pickle.load(f)
    return output

# Whenever you have a list, ensure that the product's key is equal to the product's index in the list.
# This is required by TopalogluDPOptimal

num_products = 5
initial_inventory = [4, ] * num_products
prices = [1, ] * num_products
attractions = [[1, ]*num_products]
seed = 0
T = 35
title = "Uniform Initial Inventory, Utility, and Price"
num_runs = 50000

revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
    simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)

save_single_instance_data(revenue, inventory_vectors, offered_sets, cumulative_revenue, names, title)

print("First test complete")

# Second Test
title = "Uniform Initial Inventory and Price, Varied Utility"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Base of Exponential"
T = 20
for x in np.arange(1, 10.1, 0.5):
    attractions = [[x**(i-2) for i in range(num_products)]]
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Second test complete")

# Third Test
title = "Uniform Utility and Price, Varied Initial Inventory"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Initial Inventory"
attractions = [[1, ]*num_products]
T = 20
for x in np.arange(1, 10.1, 1):
    initial_inventory = [x, ]*num_products
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Third test complete")

# Fourth Test
title = "Uniform Utility and Price, Uneven Initial Inventory"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Inventory Unevenness Parameter"
T = 45
for x in np.arange(0, 4.1, 0.5):
    initial_inventory = [9 - np.floor(2*x), 9 - np.floor(x), 9, 9+np.floor(x), 9 + 2*np.floor(x)]
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Fourth test complete")

# Fifth Test
num_products = 20
initial_inventory = [20, ]*num_products
prices = [1, ]*num_products
attractions = [[1, ]*num_products]
T = 120
num_runs = 50000
title = "Large Uniform Initial Inventory, Utility, and Price"

revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
    simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)

save_single_instance_data(revenue, inventory_vectors, offered_sets, cumulative_revenue, names, title)

print("Fifth test complete")

# Sixth Test
title = "Large Uniform Initial Inventory and Price, Varied Utility"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Base of Exponential"
T = 100
initial_inventory = [5, ]*num_products
for x in np.arange(1, 1.51, 0.05):
    attractions = [[x**(i-10) for i in range(num_products)]]
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Sixth test complete")

# Seventh Test
title = "Large Uniform Utility and Price, Varied Initial Inventory"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Initial Inventory"
attractions = [[1, ]*num_products]
num_runs = 10000
for x in np.arange(1, 9.1, 1):
    initial_inventory = [x, ]*num_products
    T = 20*x
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Seventh test complete")

# Eighth Test
title = "Large Uniform Utility and Price, Uneven Initial Inventory"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Inventory Unevenness Parameter"
num_runs = 1000
for x in np.arange(0, 0.91, 0.05):
    initial_inventory = [9 - np.floor(i*x) for i in range(int(num_products/2))] + [9 + np.floor(i*x) for i in range(int(num_products/2))]
    T = sum(initial_inventory)
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Eighth test complete")

# Ninth Test
title = "Number of Products"
revenue_vs_parameter = dict()
policy_names = ""
parameter_name = "Inventory Unevenness Parameter"
num_runs = 1000
for x in np.arange(10, 30.5, 1):
    num_products = x
    initial_inventory = [5, ]*num_products
    attractions = [[1,]*num_products]
    T = sum(initial_inventory)
    (revenue, inventory_vectors, offered_sets, cumulative_revenue, names) = \
        simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
    policy_names = names
    revenue_vs_parameter[x] = revenue

save_parameter_experiment_data(revenue_vs_parameter,policy_names,parameter_name,num_runs,title)

print("Finished tests")

# plot_ratio_optimal(revenue_vs_parameter, policy_names, parameter_name, 3, tick_size) # The third policy is the clairvoyant (see simulate)

# plot_revenue_vs(revenue_vs_parameter, policy_names, parameter_name, num_runs, tick_size)

# plot_finite_difference(cumulative_revenue, num_runs, names)

# plot_cumulative_revenue(names, cumulative_revenue,num_runs)