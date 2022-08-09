import numpy as np
from scipy.sparse import *
from decimal import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
getcontext().prec = 5000

class TopalogluDPOptimal:
    """Computes the optimal sets to offer, using a dynamic programming approach outlined in the Topaloglu paper
    Assumes a single customer type.

    Attributes
    __________
    offer_sets: list
        Contains an entry for every period. For every period, contains a dictionary which has as keys every
        possible inventory vector, and as a value a set of integers, which are the product keys to offer
    initial_inventory_tuple: tuple
        Contains the initial inventory values
    """
    def __init__(self, initial_inventory, prices, utilities, np_utility, T):
        """
        Parameters
        __________
        initial_inventory: list
            the value at each index is the initial inventory of the product whose product key is equal to the index
        prices: list
            the value at each index is the price of the product whose product key is equal to the index
        utilities: list
            the value at each index is the utilities of the product whose product key is equal to the index
            where the utilities are assuming that there is a single customer type. Each should be an integer
            since the Decimal package will be applied to it
        np_utility: integer
            utility of the no purchase option for the customer in question
        """

        # vee stores the optimal revenue at each time&inventory level
        vee = []
        # offer sets stores the set of product keys to offer at each time&inventory level
        offer_sets = []
        # Both are implemented with list indices being periods and dictionaries containing the
        # inventory tuples as keys
        for x in range(T+1):
            vee += [dict()]
            if x < T:
                offer_sets += [dict()]

        num_products = len(initial_inventory)
        initial_inventory_tuple = tuple(initial_inventory)

        # DP Space contains every possible inventory vector. Generate all possible inventory vectors
        # (that is, every conbination of inventory vector where the inventory of each product is less
        # than or equal to the initial inventory of that product
        DP_space = []
        def recursion(index, vector, DP_space):
            if index == num_products:
                DP_space += [vector]
            else:
                for x in range(vector[index] + 1):
                    new_vector = tuple(vector[i] if i != index else x for i in range(len(vector)))
                    recursion(index + 1, new_vector, DP_space)
        recursion(0, initial_inventory_tuple, DP_space)

        # Initialize the recursion to generate the optimal sets and revenues
        for x in DP_space:
            vee[T][x] = Decimal('0')

        for i in range(T):
            t = T - i - 1
            for x in DP_space:
                # Compute Adjusted Revenue by DP formulation given in paper
                adjusted_revenue = [Decimal('0')]*num_products
                for i in range(num_products):
                    if x[i] > 0:
                        new_tuple = tuple(x[y] if y != i else x[i] - 1 for y in range(len(x)))
                        adjusted_revenue[i] = Decimal(prices[i]) + vee[t+1][new_tuple] - vee[t+1][x]

                # This is a permutation on range(num_products), which lists the products in order of adjusted revenue.
                adjusted_revenue_indices = sorted(range(len(adjusted_revenue)), key=adjusted_revenue.__getitem__,
                                                  reverse=True)

                # We use the same method as used for max_set, but here we use the decimal package for precision.
                numerator = Decimal('0')
                denominator = Decimal(np_utility)
                max_revenue = Decimal('0')
                final_index = -1
                for i in range(num_products):
                    index = adjusted_revenue_indices[i]
                    # If the adjusted revenue of the next largest product is smaller than the max_revenue so far,
                    # it will "bring down" the average so to speak.
                    if adjusted_revenue[index] <= max_revenue:
                        break

                    # We compute the revenue in the same
                    numerator += Decimal(utilities[index])*adjusted_revenue[index]
                    denominator += Decimal(utilities[index])
                    revenue = numerator/denominator

                    if (revenue > max_revenue):
                        max_revenue = revenue
                        final_index = i
                vee[t][x] = max_revenue + vee[t+1][x]
                offer_sets[t][x] = adjusted_revenue_indices[:(final_index+1)]
        self.offer_sets = offer_sets
        self.initial_inventory_tuple = initial_inventory_tuple

    def check_offer_everything(self, num_products):
        sets_count = 0
        for x in self.offer_sets:
            for y, z in x.items():
                zero_count = 0
                for x in y:
                    if x == 0:
                        zero_count += 1
                if len(z) != num_products - zero_count:
                    sets_count += 1
                    print("Woah!")
                    print(x)
                    print(y)
                    print(z)
        if sets_count == 0:
            print("This policy is simply the OE policy.")
            return True
        else:
            print("This is not the OE policy")
            return False


    def offer_set(self, inventory, initial_inventory, t, customer):
        inventory_list = [0]*len(self.initial_inventory_tuple)
        product_key_product_dict = dict()
        for x in initial_inventory:
            product_key_product_dict[x.product_key] = x

        for x in inventory:
            inventory_list[x.product_key] = inventory[x]
        inventory_tuple = tuple(inventory_list)
        return {product_key_product_dict[x] for x in self.offer_sets[t][inventory_tuple]}

    def __str__(self):
        return "DPO"



class DPOptimal2:
    """The class will also work when there are multiple customer types, but it has to be re-run each trial
    since some problem instances will re-generate the customer sequence each run."""
    def __init__(self, problem_instance):
        vee = []
        offer_sets = []

        # Save problem instance data
        sale_horizon = problem_instance.sale_horizon
        initial_inventory = problem_instance.initial_inventory
        num_products = len(initial_inventory)

        # List, used to make a dictionary between array indices and product indices
        initial_inventory_list = [(x, initial_inventory[x]) for x in initial_inventory]
        product_index_dict = dict()
        index_product_dict = dict()
        for x in range(num_products):
            product_index_dict[x] = initial_inventory_list[x][0]
            index_product_dict[initial_inventory_list[x][0]] = x
        initial_inventory_list = [x[1] for x in initial_inventory_list]

        initial_inventory_tuple = tuple(initial_inventory_list)
        # First, we create a list, with every possible dictionary vector.
        DP_space = []
        def recursion(index, vector, DP_space):
            if index == num_products:
                DP_space += [vector]
            else:
                for x in range(vector[index] + 1):
                    new_vector = tuple(vector[i] if i != index else x for i in range(len(vector)))
                    recursion(index + 1, new_vector, DP_space)
        recursion(0, initial_inventory_tuple, DP_space)
        T = len(sale_horizon)
        for x in range(T+1):
            vee += [dict()]
            if x < T:
                offer_sets += [dict()]
        for x in DP_space:
            vee[T][x] = 0

        for i in range(T):
            t = T - i - 1
            for x in DP_space:
                adjusted_revenue = dict()
                for i in range(len(x)):
                    product = product_index_dict[i]
                    if x[i] > 0:
                        new_tuple = tuple(x[y] if y != i else x[i] - 1 for y in range(len(x)))
                        adjusted_revenue[product] = product.price + vee[t+1][new_tuple] - vee[t+1][x]
                    else:
                        adjusted_revenue[product] = 0
                adjusted_revenue_sorted = sorted(adjusted_revenue.items(), key = lambda x: -x[1])
                max_set = maximum_revenue_set(sale_horizon[t], adjusted_revenue_sorted)
                offer_sets[t][x] = max_set
                denominator = sum(sale_horizon[t].products[x] for x in max_set) + sale_horizon[t].no_purchase_utility
                vee[t][x] = vee[t+1][x]+sum(adjusted_revenue[x]*sale_horizon[t].products[x]/denominator for x in max_set)

        self.offer_sets = offer_sets
        self.initial_inventory_tuple = initial_inventory_tuple
        self.product_index_dict = product_index_dict
        self.index_product_dict = index_product_dict

    def offer_set(self, inventory, initial_inventory, t, customer):
        inventory_list = [0]*len(self.initial_inventory_tuple)
        for x in inventory:
            inventory_list[self.index_product_dict[x]] = inventory[x]
        inventory_tuple = tuple(inventory_list)
        return self.offer_sets[t][inventory_tuple]

    def __str__(self):
        return "DPO"


class DPOptimal:
    def __init__(self, dimension, utilities, np_utility):
        stages = []
        decisions = dict()
        num_products = len(utilities)
        decisions[(0,)*num_products] = (0,)*num_products
        final_stage = {((0, 0, 0),)*num_products: Decimal('0')}
        stages += [final_stage]

        def feasible_offer_set(tuple_1, tuple_2):
            if len(tuple_1) != len(tuple_2):
                return False
            else:
                for i in range(len(tuple_1)):
                    if tuple_1[i] == 1 and tuple_2[i] == 0:
                        return False
                return True

        def return_optimal_decision_and_probability(the_tuple, expectation, utilities, np_utility):
            power_set = []
            for i in range(len(the_tuple) - 1):
                if len(power_set) == 0:
                    power_set = [(0,), (1,)]
                for x in range(len(power_set)):
                    power_set += [power_set[x] + (1,)]
                    power_set[x] = power_set[x] + (0,)
            set_to_times = dict()
            for i in power_set:
                if sum(i) != 0 and feasible_offer_set(i, the_tuple):
                    utilities_sum = sum(utilities[x] if i[x] == 1 else 0 for x in range(len(i))) + np_utility
                    utilities_sum = Decimal(utilities_sum)
                    np_probability = Decimal(np_utility)/utilities_sum
                    probabilities = [Decimal(utilities[x]) / utilities_sum if i[x] == 1 else 0 for x in range(len(i))]
                    # The probability that any given event will occur first is the probability of the
                    # event/probability over all events
                    conditional_probabilities = [probabilities[x] / (Decimal('1') - np_probability) for x in range(len(i))]
                    expected_wait_time = [probabilities[x] / ((Decimal('1') - np_probability) ** 2) for x in range(len(i))]
                    time = 0
                    for x in range(len(the_tuple)):
                        if the_tuple[x] > 0:
                            new_tuple = list(the_tuple)
                            new_tuple[x] -= 1
                            new_tuple = tuple(new_tuple)
                            time += (expected_wait_time[x] + conditional_probabilities[x] * expectation[new_tuple])
                    if time != 0:  # This would indicate that the probability of purchase is 0
                        set_to_times[i] = time
            min = sorted(set_to_times.items(), key=lambda x: (x[1], [-x for x in x[0]]))
            return min[0]

        for i in range(dimension):
            previous_stage = stages[i]
            new_stage = dict()
            for x, y in previous_stage.items():
                for z in range(len(x)):
                    new_x = x[z] + 1
                    new_tuple = list(x)
                    new_tuple[z] = new_x
                    new_tuple = tuple(new_tuple)
                    max = return_optimal_decision_and_probability(new_tuple, previous_stage, utilities, np_utility)
                    new_stage[new_tuple] = max[1]
                    decisions[new_tuple] = max[0]
            stages += [new_stage]
        self.tuple_dict = decisions

    def offer_set(self, inventory, initial_inventory, t, customer):
        key_product_dict = dict()
        for x in inventory:
            key_product_dict[x.product_key] = x
        inventory_state_tuple = [0] * len(initial_inventory)
        for x in inventory.keys():
            inventory_state_tuple[x.product_key] = inventory[x]
        inventory_state_tuple = tuple(inventory_state_tuple)
        offer_set = set()
        offer_tuple = self.tuple_dict[inventory_state_tuple]
        for x in range(len(initial_inventory)):
            if offer_tuple[x] == 1:
                offer_set.add(key_product_dict[x])
        return offer_set

    def __str__(self):
        return "DPO"


class OfferEverything:
    """Class implementing the offer everything policy"""

    def offer_set(self, inventory, initial_inventory, t, customer):
        """Returns the assortment of all in stock products

        Parameters
        __________
        inventory: dictionary
            contains all in stock products as keys, and inventory values as values

        initial_inventory, t, customer
            dummy parameters so that this method can be called by any policy object
        """

        return inventory.keys()

    def __repr__(self):
        return "OE"

    def __str__(self):
        return "OE"


def maximum_revenue_set(customer, adjusted_prices_sorted):
    """Returns the MNL revenue maximizing assortment

    Parameters
    __________
    customer: customer
        object, which contains a list of utilities for different products, from
        which the MNL model can be derived.
    adjusted_prices_sorted: list
        sorted in descending order by adjusted revenue. adjusted revenue is some metric, calculated
        by a policy which determines how valuable it is to sell any given product.

    Returns
    _______
    set which, by the MNL model, produces the greatest adjusted revenue
    """

    # Gallego and Topaloglu https://link.springer.com/book/10.1007/978-1-4939-9606-3 the set with the maximum
    # revenue has the form (1, ..., j), where (1,...,j,...n) are the products listed in descending order of
    # adjusted revenue. The MNL revenue of one of these sets of products (1, ..., j) is given by the expression
    # SUM_j(r_i w_i)/(w_np + SUM_j(w_i)), where w_i is utility and r_i is adjusted revenue.
    # We compute this value for each of these sets by adding r_j w_j to r_1 w_1 + ... + r_(j-1)w_(j-1)
    # (the previous numerator) to calculate the numerator for each set and w_j to w_np + w_1 + ... + w_(j-1)
    # to calculate the next denominator.

    # expected revenue denominator. We use a convention in this program that the no purchase probability is always 0
    utility_sum = 1
    # expected revenue numerator
    offer_set_adjusted_revenue = 0
    max_revenue = 0
    final_index = -1
    for i in range(len(adjusted_prices_sorted)):
        product = adjusted_prices_sorted[i]
        # If the adjusted revenue of the next product is lower than the last expected revenue, it will "bring down"
        # the average, and every next product will have less than or equal to that amount of adjusted revenue,
        # so we know we have reached the optimal set and we can break out of the loop
        if product[1] <= max_revenue:
            break
        # calculate the next denominator
        utility_sum += customer.product_attractions[product[0]]
        # calculate the next numerator
        offer_set_adjusted_revenue += product[1] * customer.product_attractions[product[0]]
        expected_revenue = offer_set_adjusted_revenue / utility_sum
        # We store the expected revenue maximizing set by storing the last product, that is, the product with
        # the lowest adjusted revenue which should still be included in the set, and then simply include all products
        # with higher adjusted revenue.
        if max_revenue < expected_revenue:
            max_revenue = expected_revenue
            final_index = i
    return {x[0] for x in adjusted_prices_sorted[:final_index + 1]}


class IBPolicy:
    """Class implementing the inventory balancing policy from paper https://doi.org/10.1287/mnsc.2014.1939

    Attributes
    __________
    balancing_function: func
        function from [0,1]-->[0,1] which reduces the effective revenue of a product if its inventory is low
    policy_name: str
        the name of the policy, considering the balancing function we are using
    """

    def __init__(self, balancing_function):
        """
        Parameters
        __________
        balancing_function: func
            function from [0,1]-->[0,1] which reduces the effective revenue of a product if its inventory is low
        policy_name: str
            the name of the policy, considering the balancing function we are using
        """
        self.balancing_function = balancing_function

    def offer_set(self, inventory, initial_inventory, t, arriving_customer_type):
        """
        Makes a product assortment decision, for which products to offer the customer arriving at period t

        Parameters
        __________
        inventory: dictionary
            contains the products with non-zero inventory as keys and the current inventory levels at period t as values
        initial_inventory: dictionary
            contains all products as keys, and the initial inventory levels for the products as values
        t:
            index of the period in the selling horizon
        sale_horizon: list
            has length of the selling horizon, with a product for each element, where an element represents a period
        """
        # Adjusted price for each product i is p_i * phi(x_i/c_i), where phi is the inventory balancing function,
        # x_i is the inventory level, c_i is the initial inventory level, and p_i is the price of the product.
        # Reduces the likelihood that the policy will offer products with comparatively low inventory.
        adjusted_prices = dict((x, self.balancing_function(inventory.get(x) / initial_inventory.get(x)) * x.price)
                               for x in inventory)
        adjusted_prices_sorted = sorted(adjusted_prices.items(), key=lambda x: (-x[1], -x[0].product_key))
        return maximum_revenue_set(arriving_customer_type[t], adjusted_prices_sorted)

    def __repr__(self):
        return "IB"

    def __str__(self):
        return "IB"


class DPAPolicy:
    """Class implementing the DPA algorithm, comes from https://doi.org/10.1287/opre.2019.1931

    Attributes
    __________
    gamma: list
        stores gamma values for each product, at each period. Recall that gamma values are a crude approximation
        of the expected total revenue that can be extracted from a product at each period. Implemented as a list with
        one entry for each time period, and at each time period there is a dictionary, with the products as keys
        and the expected total revenue as values.
    coefficient: float
        used in the calculation for the availability tracking basis function.
    theta: float
        tuning parameter used to calculate the gamma values. Adjusts the opportunity cost of losing inventory.
    """

    def __init__(self, problem_instance, theta):
        """
        Parameters
        __________
        problem_instance: DynamicAssortmentOptimizationProblem
            a problem instance represents a run of an experiment, and contains all variables related to
            the simulation, including the customer types arriving during the selling horizon, the product
            types being sold in the simulation, and the initial inventory vector
        theta: float
            tuning parameter used to calculate the gamma values. Adjusts the opportunity cost of losing inventory.
        """
        arriving_customer_type = problem_instance.arriving_customer_type
        initial_inventory = problem_instance.initial_inventory
        T = len(arriving_customer_type)

        # This is the gamma vector, which is calculated with a recursion.
        # Thus, we need one more period.
        gamma = [dict() for x in range(T + 1)]
        # Initializing the recursion
        for p in initial_inventory:
            gamma[T][p] = 0

        # Recursion step
        for i in range(T):
            # We need a variable to index backwards over the gamma list
            t = T - i - 1
            # See how gamma is calculated in the paper, section 4.1. Calculate the adjusted revenue for each product
            # and put it in a dictionary
            adjusted_prices = dict((x, (x.price - theta * gamma[t + 1][x] / initial_inventory[x])) for x in initial_inventory)
            # Compute the "expected set" which we will use to calculate the purchase probabilities
            adjusted_prices_sorted = sorted(adjusted_prices.items(), key=lambda x: (-x[1], -x[0].product_key))
            max_set = maximum_revenue_set(arriving_customer_type[t], adjusted_prices_sorted)
            # Calculate the numerator for the probability calculation
            utility_sum = sum(arriving_customer_type[t].product_attractions[p] for p in max_set)
            for p in initial_inventory:
                if p in max_set:
                    # We split up the expression for gamma in two for readability
                    arg = p.price - theta * gamma[t + 1][p] / initial_inventory[p]
                    # This is the expression for the gamma recursion
                    gamma[t][p] = (arriving_customer_type[t].product_attractions[p] / utility_sum) * arg + gamma[t + 1][p]
                else:
                    gamma[t][p] = gamma[t + 1][p]
        self.gamma = gamma
        self.coefficient = 1 / (1 - np.exp(-1))
        self.theta = theta

    def function(self, x):
        """Instead of re-computing self.coefficient every time this calculation is run,
        we save it as an attribute. It saves a surprising amount of time since this function gets called so much"""
        return (1 - np.exp(-x)) * self.coefficient

    def basis_function(self, product, inventory, initial_inventory):
        """Function which tracks the availability of a product based on the ratio x between its
        inventory and its starting inventory."""
        return self.function(inventory.get(product) / initial_inventory.get(product))

    def offer_set(self, inventory, initial_inventory, t, arriving_customer_type):
        """Makes a product assortment decision, for which products to offer the customer arriving at period t

        Parameters
        __________
        inventory: dictionary
            contains the products with non-zero inventory as keys and the current inventory levels at period t as values
        initial_inventory: dictionary
            contains all products as keys, and the initial inventory levels for the products as values
        t:
            index of the period in the selling horizon
        sale_horizon: list
            has length of the selling horizon, with a product for each element, where an element represents a period
        """
        adjusted_prices = dict()
        # Compute the adjusted revenue for each product, using the expression from the paper: r_i - H_f - H_i
        # In the case where each product uses one unique resource,
        # r_i + H_f - H_i = r_i - gamma(t+1)[x](phi(x_f)-phi(x_i))
        for x in inventory:
            if inventory[x] == 0:  # This line of code should never run because unused inventory gets removed
                adjusted_prices[x] = 0
            else:
                final_inventory = inventory.copy()
                final_inventory[x] = inventory[x] - 1
                adjusted_prices[x] = x.price - self.gamma[t + 1].get(x) * self.basis_function(x, inventory,
                                                                                              initial_inventory) + \
                                     self.gamma[t + 1].get(x) * self.basis_function(x, final_inventory,
                                                                                    initial_inventory)
        adjusted_prices_sorted = sorted(adjusted_prices.items(), key=lambda x: (-x[1], -x[0].product_key))
        return maximum_revenue_set(arriving_customer_type[t], adjusted_prices_sorted)

    def __str__(self):
        return "DPA"


class Clairvoyant:
    """ Class implementing the clairvoyant policy, which knows the decisions the customers will make in advance

    Attributes
    __________
    offer_set: list
        with the same length as the selling horizon, and each element contains a set to offer the customers each period
    """
    def __init__(self, problem_instance):
        """
        Parameters
        __________
        problem_instance: DynamicAssortmentOptimizationProblem
            contains attributes:
                product_utilities: list
                    where each element is a list of ordered pairs, where the first element is a product and the second
                    element of the ordered pair is the utility value for the arriving customer. The list is in descending
                    order of utility
                no_purchase_utilities: list
                    where each element is the utility of not making a purchase, each period
                initial_inventory: dictionary
                    where the keys are the products and the values are the initial inventory values
        """
        product_utilities = problem_instance.product_utilities
        no_purchase_utilities = problem_instance.no_purchase_utilities
        initial_inventory = problem_instance.initial_inventory
        # The scipy sparse matrix stores each product as an index. This dictionary lets us access the product
        # afterwards with each of the indices
        key_product_dict = {p.product_key:p for p in initial_inventory}
        T = len(product_utilities)
        num_products = len(initial_inventory)

        # Each arriving customer has a set of products he is willing to buy: the products whose utilities crystallized
        # to a value higher than the value the no purchase utility crystallized to.
        # In the paper, the customers are grouped by type, and assigned a population, with one node per type
        # in the graph. However, we keep them separate so that we can see which product gets sold each period, and
        # directly compare this information with the other policies.
        customers = []
        for t in range(T):
            customer = []
            i = 0
            while i < num_products and product_utilities[t][i][1] > no_purchase_utilities[t]:
                customer += [product_utilities[t][i][0].product_key]
                i += 1
            customers += [customer]

        # In the offer everything paper, each product, and customer, has a node. There is a node from a customer
        # to a product as long as the utility of that product is higher than the no purchase utility, that is,
        # if the customer is potentially willing to buy the product if no other more favourable products are available.
        # Now, in the matrix, each node is represented by an index. If there are m customers, we let the node indices
        # of the customers be 0, ..., m-1. For the n products, we let m, ..., m+n-1 be the node indices of products
        # These dictionaries allow us to access the product keys from the node indices, and vice versa.
        product_node_index = dict()
        node_index_product = dict()

        # There are T customers, so we assign node indices T to T+n to the products.
        i = T
        for p in initial_inventory:
            product_node_index[p.product_key] = i
            node_index_product[i] = p
            i += 1

        # The flow problem will have a single sink and a single source. The source is connected to the
        # customers, the customers are connected to the products which they are willing to buy,
        # and the sink is connected to the products. A customer buys a product if there is some flow
        # going from that customer to a product. Each customer can only buy one product, so there is an
        # arc capacity limit of 1 from the source node to each customer node.
        source_index = i
        sink_index = i + 1

        # All three of these lists will have the same length. For any index i, the rows list entry i
        # is the index of the source node of the edge, the column list entry contains the index of the
        # sink node, and the data list entry i contains the arc capacity for that edge.
        na_matrix_rows = []
        na_matrix_cols = []
        na_matrix_data = []

        # generate the edges as described
        for t in range(T):
            for p in customers[t]:
                na_matrix_rows += [t]
                na_matrix_cols += [product_node_index[p]]
                na_matrix_data += [1]

        for t in range(T):
            na_matrix_rows += [source_index]
            na_matrix_cols += [t]
            na_matrix_data += [1]

        for p, x in initial_inventory.items():
            na_matrix_rows += [product_node_index[p.product_key]]
            na_matrix_cols += [sink_index]
            na_matrix_data += [x]

        # create and solve the matrix
        na_matrix = csr_matrix((na_matrix_data, (na_matrix_rows, na_matrix_cols)),
                               shape=(sink_index + 1, sink_index + 1))
        dictionary_of_keys_graph = csgraph.maximum_flow(na_matrix, source_index, sink_index).residual.todok()
        # convert the dictionary of keys scipy cs-graph format into a real dictionary
        graph_edges = dictionary_of_keys_graph.keys()
        graph_dictionary = dict()
        for k in graph_edges:
            graph_dictionary[k] = dictionary_of_keys_graph.get(k)
        # We loop through the edges. For all the edges from a given customer to products, at most one of them will
        # not be 0. We search these edges, and if one of them has flow 1, then it means that the optimal solution
        # has the customer arriving at period t buy that product, so we add it as the sole member of the set offered
        # that period, so that the customer doesn't buy any other products instead
        offer_sets = []
        for t in range(T):
            period_t_set = set()
            for x in customers[t]:
                if graph_dictionary[(t, product_node_index[x])] == 1:
                    period_t_set.add(key_product_dict[x])
            offer_sets += [period_t_set]
        self.offer_sets = offer_sets

    def offer_set(self, inventory, initial_inventory, t, sale_horizon):
        return self.offer_sets[t]

    def __str__(self):
        return "CLV"
