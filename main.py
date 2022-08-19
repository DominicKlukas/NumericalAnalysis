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
import pandas as pd
import openpyxl

np.set_printoptions(threshold=sys.maxsize)

T = 5


# These functions generate a list with a customer objects for each period in the selling horizon as elements
def single_customer_type(customer_list, seed):
    return [customer_list[0]] * T


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


def sequentially_arriving_customer_types(customer_list, seed):
    num_customers = len(customer_list)
    segment_length = int(T / num_customers)
    final_segment = T - segment_length*(num_customers-1)
    arriving_customer_types = []
    for i in range(num_customers - 1):
        arriving_customer_types += [customer_list[i]]*segment_length
    arriving_customer_types += [customer_list[num_customers-1]]*final_segment
    return arriving_customer_types


def reverse_sequentially_arriving_customer_types(customer_list, seed):
    num_customers = len(customer_list)
    segment_length = int(T / num_customers)
    final_segment = T - segment_length*(num_customers-1)
    arriving_customer_types = []
    for i in range(num_customers - 1):
        arriving_customer_types += [customer_list[num_customers - i - 1]]*segment_length
    arriving_customer_types += [customer_list[0]]*final_segment
    return arriving_customer_types


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


def create_excel_table(policies, prices_list, inventory_list, attractions_list, sale_horizon_length_list,
                       num_runs, customer_type_generating_function, baseline_policy_index, filepath, seed):
    """
    Function that generates an exel table of results, for a set of experiments, where each row's experiments are
     run with a parameters permuted through every combination of the lists of parameters provided in the arguments.
    The exel table has a column for each
    The total revenue for one baseline policy and the ratio of the revenue between each other policy and the
    baseline revenue is displayed.

    Args:
        policies:
            A list of class objects, where each class is a policy implementation. The class gets
            instantiated in simulation.py, where it receives the required randomly generated customers
            and parameters for each trial
        prices_list:
            List of lists, where each list is the list of prices for the products.
        inventory_list:
            List of lists, where each list is the initial inventory levels for the products.
        attractions_list:
            A list of a list of a list. We permute through each list of lists. Each list of lists represents
            a set of customer types, where each customer type is given by a list of attraction values attributed
            to each product.
        sale_horizon_length_list:
            function of the initial inventory, which generates a list of selling horizon lengths to permute through
        num_runs:
            number of trials for each experiment
        customer_type_generating_function:
            function which determines how the different customer types defined in each element of attractions_list are
            distributed over the selling horizon
        baseline_policy_index:
            index of the policy in policies which the other policies should be compared to
        filepath:
            pathname of the file to output
        seed:
            Integer which uniquely determines the outcome of the simulation

    Returns:
        Generates an excel spreadsheet
    """
    data = [['Prices', 'Attractions', 'Initial Inventory', 'T', 'Baseline Policy'] + ['' for x in range(len(policies))]]
    print(data)
    for prices in prices_list:
        for attractions in attractions_list:
            for inventory in inventory_list:
                for T in sale_horizon_length_list(inventory):
                    revenue, inventory_vectors, offered_sets, cumulative_revenue, names, revenues_for_each_experiment = \
                        simulation(policies, inventory, prices, attractions, customer_type_generating_function, T, num_runs, seed)
                    data[0][-len(policies):] = names
                    data += [[tuple(prices), tuple(attractions[0]), tuple(inventory), T, revenue[0] / num_runs]
                             + [revenue[n] / revenue[baseline_policy_index] for n in range(len(policies))]]
    # convert your array into a dataframe
    df = pd.DataFrame(data)
    # save to xlsx file
    df.to_excel(filepath, index=False, header=False)


num_products = 5
initial_inventory_list = [[1, 1, 1, 1, 1], [5, 5, 5, 5, 5], [10, 10, 10, 10, 10],
                          [9, 7, 5, 3, 1], [7, 6, 5, 4, 3], [3, 4, 5, 6, 7],[1, 3, 5, 7, 9]]
prices_list = [[1, ]*num_products]
attractions_list = [[[1, ]*num_products], [[0.25, 0.5, 1, 2, 4]], [[1/9, 1/3, 1, 3, 9]]]
num_runs = 100000
seed = 0
cp_index = 0
filepath = '/Users/dominic/Desktop/table 1 results Clairvoyant.xlsx'
policies = [OfferEverything, IBPolicy, generate_dpa_class_object(1.6), Clairvoyant]

def sale_horizon_length(inventory):
    return [sum(inventory)]

create_excel_table(policies, prices_list, initial_inventory_list, attractions_list, sale_horizon_length, num_runs,
                   single_customer_type, cp_index, filepath, seed)

