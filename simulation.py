import numpy as np
from policy import *
from problem_instance import *


def psi_eib(x):
    psi_c = np.exp(1) / (np.exp(1) - 1)
    return psi_c * (1 - np.exp(-x))


def psi_myopic(x):
    if x > 0:
        return 1
    else:
        return 0


def psi_lib(x):
    return x


def simulation(initial_inventory, prices, attractions, generate_arriving_customer_types,
                                  T, num_runs, seed):

    # Use the same product objects for every single simulation
    product_list = []
    num_products = len(prices)
    for i in range(num_products):
        product_list += [Product(i, prices[i])]
    initial_inventory_dict = dict(zip(product_list, initial_inventory))

    customer_list = []
    for i in range(len(attractions)):
        product_attractions = dict(zip(product_list, attractions[i]))
        customer_list += [Customer(i, product_attractions)]

    policies = []
    policies += [OfferEverything()]
    policies += [IBPolicy(psi_eib, "Inventory Balancing")]
    policies += [None]
    # policies += [None]
#    policies += [TopalogluDPOptimal(initial_inventory, prices, attractions[0], 1, T)]

    num_policies = len(policies)

    cumulative_output = []
    for i in range(num_runs):
        arriving_customer_type = generate_arriving_customer_types(customer_list, seed+i)
        simulate = DynamicAssortmentOptimizationProblem(product_list, initial_inventory_dict,
                                                        arriving_customer_type, seed+i)
        # These policies require recalculation each time the simulation is generated
        # policies[3] = Clairvoyant(simulate)
        policies[2] = DPAPolicy(simulate, 1.6)

        print(i)
        for j in range(len(policies)):
            output = simulate.simulation(policies[j])
            if i == 0:
                cumulative_output += [output]
            else:
                # We accumulate on the 4 outputs from the problem instance:
                # revenue, inventory vectors, offered set vector, and cumulative revenue
                cumulative_output[j] = [np.add(cumulative_output[j][k],output[k]) for k in range(4)]

    revenue = [cumulative_output[i][0] for i in range(num_policies)]
    inventory_vectors = [cumulative_output[i][1] for i in range(num_policies)]
    offered_sets = [cumulative_output[i][2] for i in range(num_policies)]
    cumulative_revenue = [cumulative_output[i][3] for i in range(num_policies)]
    names = [str(policies[i]) for i in range(num_policies)]
    return revenue, inventory_vectors, offered_sets, cumulative_revenue, names
