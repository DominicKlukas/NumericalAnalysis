import numpy as np
from policy import *
from problem_instance import *
import sys
from os.path import isfile
import os
import cProfile
from image_handling import *
from graph_all_inventory_vectors import *
from simulation import *
import matplotlib.pyplot as plt
import pickle

def roundit(x):
    if x > 0:
        return int(x)
    else:
        return -1 * int(np.abs(x))


# Whenever you have a list, ensure that the product's key is equal to the product's index in the list.
np.set_printoptions(threshold=sys.maxsize)

final_revenue = dict()
# Problem Setup
num_products = 5

C_i = [4, ]*num_products
C_v = 1
prices = [1, ]*num_products
LF = 1
num_runs = 1000  # Added on to the first run, so total runs will be 1 + num_runs
seed = 0
T = 25
title = "Large Example Uneven Prices"
np_utility =1
utilities = [i+1 for i in range(num_products)]

def w_t(t, i, num_products, T):
    return 1/(1 + np.exp(5*(i - num_products*(T - t -1)/(T-1) - 1.5)))

def w_np(t):
    return 1

revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
    simulate_single_customer_type(C_i, prices, utilities, np_utility, T, num_products, num_runs, seed)


# base_result = dict()
# for a in np.arange(0, 4.5, 0.5):
#     utilities = [[1/(1 + np.exp(a*(i+1 - (k+1.5)))) for i in range(num_products)] for k in range(num_customers)]
#     np_utilities = [1/(1 + np.exp(a*(k+1.5))) for k in range(num_customers)]
#     revenue, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, names = \
#         simulate_multiple_customer_types(C_i, C_v, prices, utilities, np_utilities, T, num_products, num_customers, num_runs, seed)
#     base_result[a] = revenue

# pickle_data = revenue, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, names
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


def plot_revenue_vs(output): #Exclude the optimal policy
    base_result, names = output
    names = ["OE", "IB", "DPA", "OP"]
    domain = sorted(base_result.keys())
    print(domain)
    for i in range(len(names)-1):
        function = [base_result[x][i] / num_runs for x in domain]
        plt.plot(domain, function, label=names[i])
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(min(domain), max(domain), 0.05))
    plt.xlabel("Base of Exponential", size=10)
    plt.ylabel("Average Revenue", size=10)
    plt.title("Base of Exponential vs Average Revenue", size=15)
    plt.show()


def plot_ratio_optimal(output):
    base_result, names = output
    domain = sorted(base_result.keys())
    names = ["OE", "IB", "DPA", "OP"]
    for i in range(len(names)-1):
        function = [base_result[x][i]/base_result[x][2] for x in domain]
        plt.plot(domain, function, label=names[i])
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(min(domain), max(domain), 0.05))
    plt.xlabel("Base of Exponential", size=10)
    plt.ylabel("Performance Ratio", size=10)
    plt.title("Base of Exponential vs Performance Ratio", size=15)
    plt.show()

# plot_ratio_optimal(output)
#
# plot_revenue_vs(output)


def plot_finite_difference(cumulative_revenue, num_runs, names):
    differences = []
    colors = ['C3', 'C4', 'C2', 'C1', 'C4']
    fig = plt.figure()
    for x in range(len(names)):
        differences += [
            [(cumulative_revenue[x][t + 1] - cumulative_revenue[x][t])/num_runs for t in range(len(cumulative_revenue[x]) - 1)]]
        plt.plot(np.arange(1, len(differences[x]) + 1, 1), differences[x], label=names[x], color=colors[x])
    plt.legend(prop={'size':12})
    plt.grid()
    plt.title("Average Revenue Per Period", size=25)
    plt.ylabel("Revenue", size=15)
    plt.xlabel("Period", size=15)
    plt.xticks([1, ] + list(np.arange(5, len(differences[0]) + 2, 5)))
    fig.set_figwidth(15)
    fig.set_figheight(7.25)
    plt.show()

# plot_finite_difference(cumulative_revenue, num_runs, names)

def plot_cumulative_revenue(policy_names, cumulative_revenue,num_runs):
    max_value = np.max([np.argmax(cumulative_revenue[x]) for x in range(len(policy_names))])
    fig = plt.figure()
    colors = ['C3', 'C4', 'C2', 'C1', 'C4']
    for i in range(len(policy_names)):
        plt.plot(cumulative_revenue[i]/num_runs, label=policy_names[i], color=colors[i])
    plt.title("Average Revenue vs Period", pad=20, size=25)
    plt.xlabel("Period", size=15)
    plt.ylabel("Average Revenue", size=15)
    plt.legend(prop={'size': 12})
    plt.grid()
    fig.set_figwidth(15)
    fig.set_figheight(7.25)
    plt.show()

plot_cumulative_revenue(names, cumulative_revenue,num_runs)

# graph_plots(inventory_vectors, names, offered_sets, cumulative_revenue)

# create_gif(names, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, 10, num_runs)
