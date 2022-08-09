import numpy as np
from product import Product
from customer import Customer
import matplotlib.pyplot as plt


class DynamicAssortmentOptimizationProblem:
    """Class representing a problem instance, for a dynamic assortment optimization problem
    Attributes
    __________
    product_utilities: list
        with the length of the selling horizon. Each element is a list of ordered pairs, (product, utility)
        where the utilities are generated randomly according to the MNL model. The list is sorted in descending
        order of utility, so that the first element in the list contains the product with the highest utility.
    no_purchase_utilities: list
        with length of the selling horizon, and floats as elements, which are the utilities of the no purchase
        option for the customer arriving that period.
    initial_inventory: dict
        with the products as keys and the initial inventory levels as values
    arriving_customer_type: list
        containing a customer object for each period in the selling horizon
    """

    def __init__(self, product_list, initial_inventory, arriving_customer_type, seed):
        """
        Parameters
        __________
        product_list: list
            of all product objects present in the problem instance
        initial_inventory: dict
            with the products as keys and the initial inventory levels as values
        arriving_customer_type: list
            containing a customer object for each period in the selling horizon
        seed: integer
            which is fed into the random number generator, so that the results are reproducible
        """
        # In the MNL model, the variability in the customer's utility is given by a gumbel distribution, with a
        # mean of 0. The mean of an unmoved gumbel distribution is gamma*beta. (beta is the scale factor).
        # We use beta=1 in our experiments, so that the attractions of the MNL model are computed as
        # e^{FixedUtility*Beta} = e^{FixedUtility}. Then, we can simply set the FixedUtility=ln(attraction)
        # for the desired attraction value.
        location = -np.euler_gamma
        product_utilities = []
        no_purchase_utilities = []
        np.random.seed(seed)

        for customer in arriving_customer_type:
            product_utility_dict = dict()  # stores the utility values for customer arriving at period t
            for p in product_list:
                # Here, we use a phi value of 1. Check revenue management and pricing analytics by Topaloglu,
                # chapter 4.4, to see that this following expression (with phi 1) gives the desired probabilities
                # so that the attraction values match the attraction values given by the customer
                product_utility_dict[p] = np.log(customer.product_attractions[p]) + np.random.gumbel(location, 1)
            no_purchase_utilities += [np.random.gumbel(location, 1)]
            # We turn the dictionary into a sorted list, to make finding the chosen product easier
            product_utility_dict = sorted(product_utility_dict.items(), key=lambda x: (-x[1], -x[0].product_key))
            product_utilities += [product_utility_dict]

        self.product_utilities = product_utilities
        self.no_purchase_utilities = no_purchase_utilities
        self.initial_inventory = initial_inventory
        self.arriving_customer_type = arriving_customer_type

    def simulation(self, policy):
        """ Runs the simulation on the problem instance.
        Parameters
        __________
        policy: policy
            which makes the sale decisions during the simulation

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
        revenue = 0
        inventory = self.initial_inventory.copy()
        key_product_dict = {x.product_key: x for x in inventory}

        # This stuff is for graphing
        inventory_vectors = []
        num_products = len(self.initial_inventory)
        offered_set_vector = []
        cumulative_revenue = [0]
        # No longer for graphing

        T = len(self.arriving_customer_type)

        for t in range(T):
            # get the policy to make a sale decision
            offered_set = policy.offer_set(inventory, self.initial_inventory, t, self.arriving_customer_type)
            # reduce the list containing the product utilities to only contain the products offered in the set
            offered_product_utilities = [(p, v) for p, v in self.product_utilities[t] if p in offered_set]
            # find which product/no_purchase_option has the highest utility, which will be the customers decision
            if len(offered_set) == 0 or offered_product_utilities[0][1] < self.no_purchase_utilities[t]:
                product_chosen = None
            else:
                product_chosen = offered_product_utilities[0][0]

            # Store data
            inventory_vectors += [
                [inventory[key_product_dict[i]] if key_product_dict[i] in inventory else 0 for i in
                 range(num_products)]]
            offered_set_vector += [
                [1 if key_product_dict[i] in offered_set or key_product_dict[i] not in inventory
                      or self.arriving_customer_type[t].product_attractions[key_product_dict[i]] == 0 else 0 for i in
                 range(num_products)]]

            # Increment inventory vector
            if product_chosen is not None:
                revenue += product_chosen.price
                inventory.update({product_chosen: inventory.get(product_chosen) - 1})
                if inventory.get(product_chosen) == 0:  # This makes the code faster.
                    inventory.pop(product_chosen)

            # store data
            cumulative_revenue += [revenue]

        return revenue, inventory_vectors, offered_set_vector, cumulative_revenue
