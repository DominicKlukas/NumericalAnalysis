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


# Methods for Graphing


# Whenever you have a list, ensure that the product's key is equal to the product's index in the list.
# This is required by TopalogluDPOptimal

# Parameters required for a simulation
# Customer types
# Distribution of customer types over product horizon
# Number of products, with prices, and initial inventory levels
# Number of runs and seed

num_products = 5
initial_inventory = [4, ] * num_products
C_v = 1
prices = [1, ] * num_products
num_runs = 1000
attractions = [[1,]*num_products]
seed = 0
T = 40
title = "Large Example Uneven Prices"


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

revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
    simulation(initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)


# plot_finite_difference(cumulative_revenue, num_runs, names)


# plot_cumulative_revenue(names, cumulative_revenue,num_runs)


# pickle_data = revenue, inventory_vectors, offered_sets, cumulative_revenue, names
#
# filename = r"saved_data/single_instances/" + title + ".pickle"
#
# i = 0
# while isfile(filename):
#     filename = r"saved_data/single_instances/" + title + "_" + str(i) + ".pickle"
#     i += 1
#
# with open(filename, 'wb') as handle:
#     pickle.dump(pickle_data, handle)
#
# with open(r"saved_data/single_instances/" + title + ".pickle", 'rb') as f:
#     output = pickle.load(f)

# base_result = dict()
# for x in np.arange(-5, 6, 1.0):
#     np_utility = 2**x
#     (revenue, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, names) = \
#         simulate_single_customer_type(C_i, prices, utilities, np_utility, T, num_products, num_runs, seed)
#     base_result[x] = revenue


# pickle_data = base_result, names
#
# filename = r"saved_data/" + title + ".pickle"
#
# i = 0
# while isfile(filename):
#     filename = r"saved_data/" + title + "_" + str(i) + ".pickle"
#     i += 1
#
# with open(filename, 'wb') as handle:
#     pickle.dump(pickle_data, handle)


with open(r"saved_data/" + title + ".pickle", 'rb') as f:
    output = pickle.load(f)


# plot_ratio_optimal(output)


# plot_revenue_vs(output)



# graph_inventory_plots(inventory_vectors, names, offered_sets, cumulative_revenue)
