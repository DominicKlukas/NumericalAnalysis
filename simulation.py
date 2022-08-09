import numpy as np
from policy import *
from problem_instance import *


# This is for the IB policy, which has its balancing functino as an argument
def psi_eib(x):
    psi_c = np.exp(1) / (np.exp(1) - 1)
    return psi_c * (1 - np.exp(-x))


def simulation(initial_inventory, prices, attractions, generate_arriving_customer_types, T, num_runs, seed):
    """
    Function that runs multiple problem instances, and produces cumulative results over all of the problem instance
    simulations.

    Parameters
    __________
    initial_inventory: list
        for each index, gives the initial inventory value for the product with the same product index
    prices: list
        for each index, gives the price of the product with the same product index
    attractions: list
        for each customer type, gives a list of attraction values for each product index
    generate_arriving_customer_types: function
        which populates a list of length selling horizon (T) with the customer types
    T: integer
        length of the selling horizon
    num_runs: integer
        number of problem instances over which results are accumulated
    seed: number
        which is fed into the random number generators as seeds, to make simulations deterministic

    Returns
    _______
    revenue: integer
        containing the revenue earned by the policy over the selling horizon
    inventory_vectors: list
        containing a list of inventory values for each period
    offered_set_vector: list
        containing a list, with the length of the number of products for each period. This list contains
        a 0 for each product key where the product is withheld from the customer despite being available, and
        a 1 otherwise.
    cumulative_revenue: list
        for each period, contains the revenue earned up to that period
    """
    # generate the product objects (the same ones will be used for every simulation)
    product_list = []
    num_products = len(prices)
    for i in range(num_products):
        product_list += [Product(i, prices[i])]
    initial_inventory_dict = dict(zip(product_list, initial_inventory))

    # generate the customer objects (the same ones will be used for every simulation)
    customer_list = []
    for i in range(len(attractions)):
        product_attractions = dict(zip(product_list, attractions[i]))
        customer_list += [Customer(i, product_attractions)]

    # the policies tested in the experiment
    policies = []
    policies += [OfferEverything()]
    policies += [IBPolicy(psi_eib)]
    policies += [None]
    policies += [None]
#    policies += [TopalogluDPOptimal(initial_inventory, prices, attractions[0], 1, T)]

    num_policies = len(policies)

    cumulative_output = []
    for i in range(num_runs):
        arriving_customer_type = generate_arriving_customer_types(customer_list, seed+i)
        simulate = DynamicAssortmentOptimizationProblem(product_list, initial_inventory_dict,
                                                        arriving_customer_type, seed+i)
        # These policies require recalculation each time the simulation is generated
        policies[2] = DPAPolicy(simulate, 1.6)
        policies[3] = Clairvoyant(simulate)

        # print(i)
        for j in range(len(policies)):
            output = simulate.simulation(policies[j])
            if i == 0:
                # np.add can only be used with an array that is already initialized
                cumulative_output += [output]
            else:
                # We accumulate on the 4 outputs from the problem instance:
                # revenue, inventory vectors, offered set vector, and cumulative revenue
                # from the current run to all of the previous runs
                cumulative_output[j] = [np.add(cumulative_output[j][k],output[k]) for k in range(4)]

    revenue = [cumulative_output[i][0] for i in range(num_policies)]
    inventory_vectors = [cumulative_output[i][1] for i in range(num_policies)]
    offered_sets = [cumulative_output[i][2] for i in range(num_policies)]
    cumulative_revenue = [cumulative_output[i][3] for i in range(num_policies)]
    names = [str(policies[i]) for i in range(num_policies)]
    return revenue, inventory_vectors, offered_sets, cumulative_revenue, names
