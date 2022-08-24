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
# they are to be passed as an argument into the simulation function in simulation.py
# ----------------------------------------------------------------------------------------------------------
def single_customer_type(customer_list, seed, T):
    return [customer_list[0]] * T


def randomized_multiple_customer_types(customer_list, variance, T, seed):
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

    def return_function(customer_list, seed, T):
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


def sequentially_arriving_customer_types(customer_list, seed, T):
    num_customers = len(customer_list)
    segment_length = int(T / num_customers)
    final_segment = T - segment_length*(num_customers-1)
    arriving_customer_types = []
    for i in range(num_customers - 1):
        arriving_customer_types += [customer_list[i]]*segment_length
    arriving_customer_types += [customer_list[num_customers-1]]*final_segment
    return arriving_customer_types


def reverse_sequentially_arriving_customer_types(customer_list, seed, T):
    num_customers = len(customer_list)
    segment_length = int(T / num_customers)
    final_segment = T - segment_length*(num_customers-1)
    arriving_customer_types = []
    for i in range(num_customers - 1):
        arriving_customer_types += [customer_list[num_customers - i - 1]]*segment_length
    arriving_customer_types += [customer_list[0]]*final_segment
    return arriving_customer_types
# ----------------------------------------------------------------------------------------------------------


# These functions save and open data as pickle files, so trials which have a long computing times only need
# to be run once
# ---------------------------------------------------------------------------------------------------------
def save_single_instance_data(revenue, inventory_vectors, offered_sets, cumulative_revenue, names, title):
    # save the data as recieved by the simulation
    pickle_data = revenue, inventory_vectors, offered_sets, cumulative_revenue, names

    directory = r"experiment_data/single_instances/"
    filename = directory + title + ".pickle"

    # Check to ensure we are not overwriting data. If we are, rename the file
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
# ---------------------------------------------------------------------------------------------------------


# Here is a zoo of varied examples.
# You can uncomment an example by highlighting the code and pressing ctrl + / on windows/linux, or cmd + / on mac.

# Examples
# ---------------------------------------------------------------------------------------------------------

# Basic Examples
# -----------------

# # Single Customer Type. Prices must be positive integers.
# num_products = 5
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# inventory = [5, ]*num_products
# prices = [1, ]*num_products # Prices must be integers
# # attractions is a list of customer types, where the customer type is a list of product attraction values
# # The nested list here represents a single customer type with its list of attractions
# attractions = [[1, ]*num_products]
# T = sum(inventory)
# seed = 0
# num_runs = 1000
# revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#     simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#
# print("Single Customer Type Basic Example")
# print("--------------------------------------")
# for x in range(len(revenue)):
#     print(names[x] + ": " + str(revenue[x]))
# print("--------------------------------------")
#
#
# # Single Customer Type. Varying Parameters.
# num_products = 5
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# inventory = [5 + (i - num_products/2) for i in range(num_products)]
# prices = [i + 1 for i in range(num_products)] # Prices must be integers
# # attractions is a list of customer types, where the customer type is represented by a list of product attraction values
# attractions = [[2**(i - num_products/2) for i in range(num_products)]]
# T = sum(inventory)
# seed = 0
# num_runs = 1000
# revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#     simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#
# print("Single Customer Type Varied Parameters")
# print("--------------------------------------")
# for x in range(len(revenue)):
#     print(names[x] + ": " + str(revenue[x]))
# print("--------------------------------------")
#
# # Multiple Customer types. Computing DPOptimal for this example takes a very long time. The customer
# # arrival sequence changes every problem instance, so the policy has to recalculate the DP space
# num_products = 5
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# inventory = [5, ]*num_products
# prices = [1, ]*num_products # Prices must be integers
# # attractions is a list of customer types, where the customer type is represented by a list of product attraction values
# attractions = [[1 if j <= i else 0.01 for j in range(num_products)] for i in range(num_products)] # Nested Customer types
# T = sum(inventory)
# seed = 0
# num_runs = 10
# # TODO: randomized_multiple_customer_types requires parameters, unlike the other cases. See function docstring.
# revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#     simulation(policies, inventory, prices, attractions,
#                randomized_multiple_customer_types(attractions, 1, T, seed), T, num_runs, seed)
#
# print("Multiple Customer Type Random Arrivals")
# print("--------------------------------------")
# for x in range(len(revenue)):
#     print(names[x] + ": " + str(revenue[x]))
# print("--------------------------------------")
#
# # Multiple Customer types, sequentially arriving customer types
# num_products = 5
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# inventory = [5, ]*num_products
# prices = [1, ]*num_products # Prices must be integers
# # attractions is a list of customer types, where the customer type is represented by a list of product attraction values
# attractions = [[1 if j <= i + 1 else 0.01 for j in range(num_products)] for i in range(num_products)] # Nested Customer types
# T = sum(inventory)
# seed = 0
# num_runs = 1000
#
# revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#     simulation(policies, inventory, prices, attractions,
#                reverse_sequentially_arriving_customer_types, T, num_runs, seed)
#
# print("Multiple Customer Type Sequential Arrivals")
# print("--------------------------------------")
# for x in range(len(revenue)):
#     print(names[x] + ": " + str(revenue[x]))
# print("--------------------------------------")


# Spreadsheet Example
# -----------------

# We create lists for each parameter, a spreadsheet which displays results for every combination of parameters
# in these lists will be generated.

# num_products = 5
# initial_inventory_list = [[3, 4, 5, 6, 7], [7, 6, 5, 4, 3]]
# prices_list = [[int(np.round((5**4)*(6/5)**i)) for i in range(num_products)]]+[[2, 3, 4, 5, 6]]
# # We can combine experiments with single and multiple customer types, as long as the way
# # in which the customers arrive are all the same. Attractions can never be 0, by MNL model
# attractions_list = [[[1, 1, 1, 1, 1]], [[1/9, 1/3, 1, 3, 9]],
#                     [[1 if i <= x else 0.01 for i in range(num_products)] for x in range(num_products)]]
# num_runs = 1000
# seed = 0
# filepath = 'numerical_analysis_spreadsheet.xlsx'
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal,
#             generate_topk_class_object(1), generate_topk_class_object(2), generate_topk_class_object(3),
#             generate_topk_class_object(4)]
# # cp_index is the index of the policy in the policies list that is our baseline policy
# cp_index = 4
#
#
# def sale_horizon_length(initial_inventory):
#     """Function returns a list of selling horizon lengths as a function of the initial inventory, which are all
#     tested."""
#     return [int(sum(initial_inventory)*0.8), sum(initial_inventory), int(sum(initial_inventory)*1.2)]
#
#
# create_excel_table(policies, prices_list, initial_inventory_list, attractions_list, sale_horizon_length,
#                    num_runs, reverse_sequentially_arriving_customer_types, cp_index, filepath, seed)


# Graphing Examples
# -----------------

# Box Plot Example

# num_products = 5
# initial_inventory_list = [[1, 3, 5, 7, 9], [3, 4, 5, 6, 7], [5, ]*num_products, [7, 6, 5, 4, 3], [9, 7, 5, 3, 1]]
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# num_runs = 1
# num_comparisons = 1000
# policies = [OfferEverything, Clairvoyant]
# prices_list = [[1, ]* num_products]
#
# x_labels = ['A', 'B', 'C', 'D', 'E']
# offer_everything_revenues = []
# clairvoyant_revenues = []
# for initial_inventory in initial_inventory_list:
#     T = sum(initial_inventory)
#     offer_everything_revenues += [[]]
#     clairvoyant_revenues += [[]]
#     for seed in range(num_comparisons):
#         revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#             simulation(policies, initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#         offer_everything_revenues[-1] += [revenue[0]]
#         clairvoyant_revenues[-1] += [revenue[1]]
#
# plot_box_graphs(clairvoyant_revenues, offer_everything_revenues, x_labels)


# Comparison between policies bar graph example
# num_products = 5
# initial_inventory_list = [[1, 3, 5, 7, 9], [3, 4, 5, 6, 7], [5, ]*num_products, [7, 6, 5, 4, 3], [9, 7, 5, 3, 1]]
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# num_runs = 1000
# seed = 0
# # Put the policy you want to compare the others to at index 0
# policies = [DPOptimal, OfferEverything, IBPolicy, generate_dpa_class_object(1.6)]
# prices_list = [[1, ]* num_products]
# x_labels = ["A", "B", "C", "D", "E"]
# comparison_revenue = []
# policy_revenues = []
# policy_names = []
# for initial_inventory in initial_inventory_list:
#     T = sum(initial_inventory)
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#     comparison_revenue += [revenue[0]]
#     policy_revenues += [revenue[1:]]
#     policy_names = names[1:]
# colors = ["firebrick", "mediumseagreen", "cornflowerblue"]
# patterns = [None, "/", "."]
# plot_comparison_charts(comparison_revenue, x_labels, policy_revenues, policy_names,
#                        "Revenue/DPO Revenue", colors, patterns)


# Comparison chart,
# num_products = 5
# initial_inventory_list = [[1, 3, 5, 7, 9], [3, 4, 5, 6, 7], [5, ]*num_products, [7, 6, 5, 4, 3], [9, 7, 5, 3, 1]]
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# num_runs = 1000
# seed = 0
# # Put the policy you want to compare the others to first
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6)]
# prices_list = [[1, ]* num_products]
# x_labels = ["A", "B", "C", "D", "E"]
# comparison_revenue = []
# policy_revenues = []
# policy_names = []
# for initial_inventory in initial_inventory_list:
#     T = sum(initial_inventory)
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, initial_inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#     comparison_revenue += [revenue[0]]
#     policy_revenues += [revenue[1:]]
#     policy_names = names[1:]
# colors = ["mediumseagreen", "cornflowerblue"]
# patterns = ["/", "."]
# plot_comparison_charts(comparison_revenue, x_labels, policy_revenues, policy_names, "Revenue/OE Revenue", colors, patterns)

# Graph Cumulative Revenue, and Revenue Per Period Example
# num_products = 5
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# inventory = [5, ]*num_products
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# T = sum(inventory)
# seed = 0
# num_runs = 100000
# revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#     simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#
# plot_cumulative_revenue(names, cumulative_revenue, num_runs)
#
# plot_revenue_per_period(cumulative_revenue, num_runs, names)


# Graph initial inventory vector
# num_products = 5
# # the graph looks better with fewer policies, although more can be added
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# inventory = [5, ]*num_products
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# T = sum(inventory)
# seed = 0
# # The graphing function only works when there is a single run
# num_runs = 1
#
# revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#     simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
# graph_inventory_plots(inventory_vectors, names, offered_sets, cumulative_revenue)

# Graphing Parameters
# -------------------

# Theta values for DPA policy
# num_products = 5
# inventory = [5, ]*num_products
# prices = [1, ]*num_products
# attractions = [[1 if j <= i else 0.01 for j in range(num_products)] for i in range(num_products)]
# T = sum(inventory)
# seed = 0
# num_runs = 1000
# revenue_vs_parameter = dict()
# policy_names = []
# tick_size = 0.2
# for theta in np.arange(1.0, 3.0, tick_size):
#     policies = [generate_dpa_class_object(theta)]
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, inventory, prices, attractions,
#                    randomized_multiple_customer_types(attractions, 1.0, T, seed), T, num_runs, seed)
#     revenue_vs_parameter[theta] = revenue
#     policy_names = names
# print(revenue_vs_parameter)
# plot_revenue_vs_parameter(revenue_vs_parameter, policy_names, "Theta", num_runs, tick_size)


# Another theta values for DPA policy example
# num_products = 5
# inventory = [5, ]*num_products
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# T = sum(inventory)
# seed = 0
# num_runs = 1000
# revenue_vs_parameter = dict()
# policy_names = []
# tick_size = 0.2
# for theta in np.arange(1.0, 3.0, tick_size):
#     policies = [generate_dpa_class_object(theta)]
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#     revenue_vs_parameter[theta] = revenue
#     policy_names = names
# print(revenue_vs_parameter)
# plot_revenue_vs_parameter(revenue_vs_parameter, policy_names, "Theta", num_runs, tick_size)


# Varying the selling horizon length. This is different than the cumulative revenue graph, since
# the DPA policy and the CLV policy are recalculated very every sale horizon graph
# num_products = 5
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant, DPOptimal]
# cp_index = 4
# inventory = [5, ]*num_products
# prices = [1, ]*num_products
# attractions = [[1, ]*num_products]
# seed = 0
# num_runs = 1000
# revenue_vs_parameter = dict()
# policy_names = []
# tick_size = 5
# for T in np.arange(0, 40, 1):
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#     revenue_vs_parameter[T] = revenue
#     policy_names = names
# plot_revenue_vs_parameter(revenue_vs_parameter, policy_names, "Selling Horizon Length", num_runs, tick_size)
# plot_ratio_optimal_vs_parameter(revenue_vs_parameter, policy_names, "Selling Horizon Length",
#                                 cp_index, tick_size)


# Two more examples: Varying the initial inventory, and the number of products.
# Notice how other parameters are recalculated
# We omit DPOptimal because it is really slow when the size of the inventory vectors increases
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant]
# cp_index = 0
# seed = 0
# num_runs = 100
# revenue_vs_parameter = dict()
# policy_names = []
# tick_size = 5
# for num_products in np.arange(0, 30, 1):
#     inventory = [5, ]*num_products
#     prices = [1, ]*num_products
#     attractions = [[1, ]*num_products]
#     T = sum(inventory)
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#     revenue_vs_parameter[num_products] = revenue
#     policy_names = names
# plot_revenue_vs_parameter(revenue_vs_parameter, policy_names, "Number of Products", num_runs, tick_size)
# plot_ratio_optimal_vs_parameter(revenue_vs_parameter, policy_names, "Number of Products", cp_index,
#                                 tick_size)

# Vary the initial inventory
# policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant]
# cp_index = 0
# seed = 0
# num_runs = 100
# num_products = 5
# prices = [1, ] * num_products
# attractions = [[1, ] * num_products]
# revenue_vs_parameter = dict()
# policy_names = []
# tick_size = 5
# for x in np.arange(0, 30, 1):
#     inventory = [x, ]*num_products
#     T = sum(inventory)
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
#         simulation(policies, inventory, prices, attractions, single_customer_type, T, num_runs, seed)
#     revenue_vs_parameter[x] = revenue
#     policy_names = names
# plot_revenue_vs_parameter(revenue_vs_parameter, policy_names, "Initial Inventory Level", num_runs, tick_size)
# plot_ratio_optimal_vs_parameter(revenue_vs_parameter, policy_names, "Initial Inventory Level", cp_index,
#                                 tick_size)
