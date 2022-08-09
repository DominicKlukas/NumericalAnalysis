import numpy as np
from product import Product
from customer import Customer
import matplotlib.pyplot as plt


class DynamicAssortmentOptimizationProblem:
    def __init__(self, product_list, initial_inventory, arriving_customer_type, seed):
        location = -np.euler_gamma
        np.random.seed(seed)
        product_utilities = []
        no_purchase_utilities = []
        for customer in arriving_customer_type:
            product_utility_dict = dict() # stores the utility values for customer arriving at period t
            for p in product_list:
                # Here, we use a phi value of 1. Check revenue management and pricing analytics by topaloglu,
                # chapter 4.4, to see that this following expression (with phi 1) gives the desired probabilities
                # so that the attraction values match the attraction values given by the customer
                product_utility_dict[p] = np.log(customer.product_attractions[p]) + np.random.gumbel(location, 1)
            no_purchase_utilities += [np.random.gumbel(location, 1)]
            product_utility_dict = sorted(product_utility_dict.items(), key=lambda x: (-x[1], -x[0].product_key))
            product_utilities += [product_utility_dict]
        self.product_utilities = product_utilities
        self.no_purchase_utilities = no_purchase_utilities
        self.initial_inventory = initial_inventory
        self.arriving_customer_type = arriving_customer_type

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

        T = len(self.arriving_customer_type)

        for t in range(T):
            offered_set = policy.offer_set(inventory, self.initial_inventory, t, self.arriving_customer_type)
            offered_product_utilities = [(p, v) for p,v in self.product_utilities[t] if p in offered_set]
            if len(offered_set) == 0 or offered_product_utilities[0][1] < self.no_purchase_utilities[t]:
                product_chosen = None
            else:
                product_chosen = offered_product_utilities[0][0]


            # This stuff is for graphing
            inventory_vectors += [
                [inventory[key_product_dict[i]] if key_product_dict[i] in inventory else 0 for i in
                 range(num_products)]]
            offered_set_vector += [
                [1 if key_product_dict[i] in offered_set or key_product_dict[i] not in inventory
                      or self.arriving_customer_type[t].product_attractions[key_product_dict[i]] == 0 else 0 for i in range(num_products)]]

            if product_chosen is not None:
                revenue += product_chosen.price
                inventory.update({product_chosen: inventory.get(product_chosen) - 1})
                if inventory.get(product_chosen) == 0:  # This makes the code faster.
                   inventory.pop(product_chosen)

            cumulative_revenue += [revenue]

        return revenue, inventory_vectors, offered_set_vector, cumulative_revenue
