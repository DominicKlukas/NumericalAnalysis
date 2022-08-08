import numpy as np
from product import Product
from customer import Customer
import matplotlib.pyplot as plt


def dirichlet_list(dimension, variance, length):
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
    a = (1 / dimension) * ((dimension - 1) / variance ** 2 - 1)
    alpha = (a,) * dimension
    c = np.random.dirichlet(alpha, 1)[0]
    # we have to scale the dirichlet distribution to the length of the list. Each entry in the dirichlet vector
    # is the number of times that the index of that entry shows up in the final list. We round the output, since
    # this must be an integer.
    c *= length
    c = np.round(c).astype(int)
    # as a result of the rounding, the parameters may no longer sum to length of the list, so we adjust
    while np.sum(c) > length:
        c[np.argmax(c)] -= 1
    while np.sum(c) < length:
        c[np.argmin(c)] += 1
    z = []
    y = 0
    for x in c:
        z += [y] * x
        y += 1
    # After putting the correct quantities of each integer in a list, we permute the integers
    z = np.random.permutation(z)
    return z.astype(int)


class DynamicAssortmentOptimizationProblem:
    def __init__(self, products, initial_inventory, sale_horizon, customer_decision_seed):
        self.products = products
        self.initial_inventory = initial_inventory
        self.sale_horizon = sale_horizon
        self.customer_decision_seed = customer_decision_seed

    def simulation(self, policy):
        revenue = 0
        inventory = self.initial_inventory.copy()

        key_product_dict = dict()
        for x in inventory:
            key_product_dict[x.product_key] = x

        # This stuff is for graphing
        inventory_vectors = []
        num_products = len(self.initial_inventory)
        offered_set_vector = []
        cumulative_revenue = [0]
        # No longer for graphing

        for t in range(len(self.sale_horizon)):
            offered_set = policy.offer_set(inventory, self.initial_inventory, t, self.sale_horizon)
            product_chosen = self.sale_horizon[t].customer_decision(offered_set, self.customer_decision_seed[t])

            # This stuff is for graphing
            inventory_vectors += [
                [inventory[key_product_dict[i]] if key_product_dict[i] in inventory else 0 for i in
                 range(num_products)]]
            offered_set_vector += [
                [1 if key_product_dict[i] in offered_set or key_product_dict[i] not in inventory
                      or self.sale_horizon[t].products[key_product_dict[i]] == 0 else 0 for i in range(num_products)]]

            if product_chosen is not None:
                revenue += product_chosen.price
                inventory.update({product_chosen: inventory.get(product_chosen) - 1})
                if inventory.get(product_chosen) == 0:  # This makes the code faster.
                   inventory.pop(product_chosen)

            cumulative_revenue += [revenue]

        # おわり
        return revenue, inventory_vectors, offered_set_vector, cumulative_revenue


class SingleCustomerType(DynamicAssortmentOptimizationProblem):
    """Single Customer Type"""
    def __init__(self, T, num_products, np_utility, C_i, utilities, prices, seed):
        # Initialize the products, setting the product prices and initial inventory capacities
        products = set()
        for i in range(num_products):
            products.add(Product(i, prices[i]))
        product_list = sorted(products)
        initial_inventory = dict()
        k = 0
        for x in product_list:
            initial_inventory[x] = C_i[k]
            k += 1

        # Use the customer types to initialize the
        product_utility_dict = dict(zip(product_list, utilities))
        customer = Customer(product_utility_dict, np_utility, 0)
        sale_horizon = [customer] * T

        location = -np.euler_gamma
        np.random.seed(seed)
        customer_decision_seed = [np.random.gumbel(location, 1, num_products + 1) for i in range(T)]
        super().__init__(products, initial_inventory, sale_horizon, customer_decision_seed)

    def __str__(self):
        return "Single Customer Type"

class MultipleCustomerTypes(DynamicAssortmentOptimizationProblem):
    """Multiple Customer Types"""
    def __init__(self, T, num_products, num_customers, C_i, C_v, np_utilities, utilities, prices, seed):
        products = set()
        for i in range(num_products):
            products.add(Product(i, prices[i]))
        product_list = sorted(products)
        initial_inventory = dict()
        k = 0
        for x in product_list:
            initial_inventory[x] = C_i[k]
            k += 1

        customers = set()

        for i in range(num_customers):
            product_weight_dict = dict(zip(product_list, utilities[i]))
            customers.add(Customer(product_weight_dict, np_utilities[i], i))  # Virtually zero no purchase probability

        np.random.seed(seed)
        dirlist = dirichlet_list(num_products, C_v, T)
        customer_list = sorted(customers)
        sale_horizon = [customer_list[x] for x in dirlist]
        customer_decision_seed = np.random.uniform(0, 1, T)
        super().__init__(products, initial_inventory, sale_horizon, customer_decision_seed)

    def __str__(self):
        return "Multiple Customer Types"

class PeriodDependentPreferences(DynamicAssortmentOptimizationProblem):
    """Multiple Customer Types"""
    def __init__(self, T, w_t, w_np, num_products, C_i, prices, seed):
        products = set()
        for i in range(num_products):
            products.add(Product(i, prices[i]))
        product_list = sorted(products)
        initial_inventory = dict()
        k = 0
        for x in product_list:
            initial_inventory[x] = C_i[k]
            k += 1

        customers = set()

        sale_horizon = []

        for t in range(T):
            utilities = [w_t(t, i, num_products, T) for i in range(num_products)]
            product_weight_dict = dict(zip(product_list, utilities))
            sale_horizon += [Customer(product_weight_dict, w_np(t), t)]

        np.random.seed(seed)
        customer_list = sorted(customers)

        customer_decision_seed = np.random.uniform(0, 1, T)
        self.C_i = C_i
        self.T = T
        self.num_products = num_products
        super().__init__(products, initial_inventory, sale_horizon, customer_decision_seed)

    def __str__(self):
        return "Multiple Customer Types"