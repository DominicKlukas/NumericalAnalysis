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


def simulate_single_customer_type(C_i, prices, utilities, np_utility, T, num_products, num_runs, seed):
    policies = []
    policies += [OfferEverything()]
    policies += [IBPolicy(psi_eib, "Inventory Balancing")]
    policies += [None]
    # policies += [TopalogluDPOptimal(C_i, prices, utilities, np_utility, T)]

    names = []
    revenue = [0]*len(policies)
    inventory_vectors = [[0]*num_products]*len(policies)
    offered_sets = [[0]*num_products]*len(policies)
    purchase_probabilities = [[0]*num_products]*len(policies)
    cumulative_revenue = [0]*len(policies)

    for i in range(num_runs):
        simulate = SingleCustomerType(T, num_products, np_utility, C_i, utilities, prices, seed + i)
        policies[2] = DPAPolicy(simulate, 1.6)
        print(i)
        for j in range(len(policies)):
            output = simulate.simulation(policies[j])
            revenue[j] = np.add(revenue[j], output[0])
            inventory_vectors[j] = np.add(inventory_vectors[j], output[1])
            offered_sets[j] = np.add(offered_sets[j], output[2])
            cumulative_revenue[j] = np.add(cumulative_revenue[j], output[3])
            purchase_probabilities[j] = np.add(purchase_probabilities[j], output[4])

    for j in range(len(policies)):
        names += [str(policies[j])]
    return revenue, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, names


def simulate_period_dependent_preferences(C_i, w_t, w_np, prices, T, num_products, num_runs, seed):
    simulate = PeriodDependentPreferences(T, w_t, w_np, num_products, C_i, prices, seed)
    policies = []
    policies += [OfferEverything()]
    policies += [IBPolicy(psi_eib, "Inventory Balancing")]
    policies += [DPAPolicy(simulate, 1.6)]

    names = []
    revenue = [0] * len(policies)
    inventory_vectors = [[0] * num_products] * len(policies)
    offered_sets = [[0] * num_products] * len(policies)
    purchase_probabilities = [[0] * num_products] * len(policies)
    cumulative_revenue = [0] * len(policies)

    for i in range(num_runs):
        simulate = PeriodDependentPreferences(T, w_t, w_np, num_products, C_i, prices, seed+i)
        policies[2] = DPAPolicy(simulate, 1.6)
        print(i)
        for j in range(len(policies)):
            output = simulate.simulation(policies[j])
            revenue[j] = np.add(revenue[j], output[0])
            inventory_vectors[j] = np.add(inventory_vectors[j], output[1])
            offered_sets[j] = np.add(offered_sets[j], output[2])
            cumulative_revenue[j] = np.add(cumulative_revenue[j], output[3])
            purchase_probabilities[j] = np.add(purchase_probabilities[j], output[4])

    for j in range(len(policies)):
        names += [str(policies[j])]
    return revenue, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, names


def simulate_multiple_customer_types(C_i, C_v, prices, utilities, np_utilities, T, num_products, num_customers, num_runs, seed):
    policies = []
    policies += [OfferEverything()]
    policies += [IBPolicy(psi_eib, "Inventory Balancing")]
    policies += [None]
    # policies += [None]

    names = []
    revenue = [0] * len(policies)
    inventory_vectors = [[0] * num_products] * len(policies)
    offered_sets = [[0] * num_products] * len(policies)
    purchase_probabilities = [[0] * num_products] * len(policies)
    cumulative_revenue = [0] * len(policies)

    for i in range(num_runs):
        simulate = MultipleCustomerTypes(T, num_products, num_customers, C_i, C_v, np_utilities, utilities, prices, seed+i)
        policies[2] = DPAPolicy(simulate, 1.6)
        # policies[3] = DPOptimal2(simulate)
        print(i)
        for j in range(len(policies)):
            output = simulate.simulation(policies[j])
            revenue[j] = np.add(revenue[j], output[0])
            inventory_vectors[j] = np.add(inventory_vectors[j], output[1])
            offered_sets[j] = np.add(offered_sets[j], output[2])
            cumulative_revenue[j] = np.add(cumulative_revenue[j], output[3])
            purchase_probabilities[j] = np.add(purchase_probabilities[j], output[4])

    for j in range(len(policies)):
        names += [str(policies[j])]
    return revenue, inventory_vectors, offered_sets, cumulative_revenue, purchase_probabilities, names