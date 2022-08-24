import numpy as np
from scipy.sparse import *
from decimal import *
import matplotlib.pyplot as plt
from ortools.graph.python import min_cost_flow, max_flow

getcontext().prec = 500

"""Each class in this module implements a policy. A policy must have the following features:
An __init__ method with problem_instance as its only parameter
An offer_set method, with parameters: inventory, initial_inventory, t, customer, which returns a set of products

Static methods:
customer_type_sensitive(): Returns boolean, indicating if the policy is sensitive to the order of that customers
arrive in a simulation
problem_instance_sensitive(): Returns boolean, indicating if the policy is sensitive to the random variable samples
in a simulation

If the policy requires other parameters in the __init__ method, a ''generating function'' can be written, which 
returns a class with problem_instance as its only __init__ parameter. We do this by defining a subclass that
instantiates the superclass with the fixed parameters, having the fixed parameters be parameters from the method. 
The generating function returns this subclass as an object.
"""


class DPOptimal:
    def __init__(self, problem_instance):
        """
        Parameters
        ----------
        problem_instance:
            problem_instance object, which contains all objects relevant to a single trial of a simulation.
            This class requires the initial_inventory levels, and the arriving_customer_type_list objects.
        """
        arriving_customer_type_list = problem_instance.arriving_customer_type_list
        T = len(arriving_customer_type_list)
        num_products = len(problem_instance.initial_inventory)
        key_product_dict = {p.product_key: p for p in problem_instance.initial_inventory}

        # Write initial inventory and prices data as lists with product keys used as indices
        initial_inventory = []
        prices = []
        for i in range(num_products):
            prices += [key_product_dict[i].price]
            initial_inventory += [problem_instance.initial_inventory[key_product_dict[i]]]

        # vee is the optimal dynamic program, storing the optimal revenue.
        vee = []
        # offer sets store a list of product keys to offer
        offer_sets = []
        utilities = []
        # Both are implemented with list indices being periods and dictionaries containing the
        # inventory tuples as keys
        # utilities is a list length T containing the utilities for the customer arriving during each period
        for x in range(T + 1):
            vee += [dict()]
            if x < T:
                offer_sets += [dict()]
                utilities += [[arriving_customer_type_list[x].product_attractions[key_product_dict[i]]
                               for i in range(num_products)]]

        initial_inventory_tuple = tuple(initial_inventory)

        # dp_space is the set of tuples {(x_1, ..., x_n) : x_i <= c_i}. We generate this set here with a recursion
        dp_space = []

        def generate_dp_space_recursion(index, vector, DP_space):
            if index == num_products:
                DP_space += [vector]
            else:
                for x in range(vector[index] + 1):
                    new_vector = tuple(vector[i] if i != index else x for i in range(len(vector)))
                    generate_dp_space_recursion(index + 1, new_vector, DP_space)

        generate_dp_space_recursion(0, initial_inventory_tuple, dp_space)

        # Initialize the dynamic program
        for x in dp_space:
            vee[T][x] = Decimal('0')

        for i in range(T):
            # Standard dynamic programming: start at end of selling horizon and work your way back to beginning
            t = T - i - 1
            for x in dp_space:
                # Compute Adjusted Revenue by DP formulation given in paper
                adjusted_revenue = [Decimal('0')] * num_products
                for j in range(num_products):
                    if x[j] > 0:
                        new_tuple = tuple(x[y] if y != j else x[j] - 1 for y in range(len(x)))
                        adjusted_revenue[j] = Decimal(prices[j]) + vee[t + 1][new_tuple] - vee[t + 1][x]

                # This is a permutation on range(num_products), which lists the product keys in order
                # of adjusted revenue.
                adjusted_revenue_indices = sorted(range(len(adjusted_revenue)), key=adjusted_revenue.__getitem__,
                                                  reverse=True)

                # We use the same method as used for max_set, to compute the optimal offer set at each period and
                # inventory level, but here we use the decimal package for precision. (See max_set)
                numerator = Decimal('0')
                # 1 is the weight of the no-purchase option
                denominator = Decimal('1')
                max_revenue = Decimal('0')
                final_index = -1
                for j in range(num_products):
                    index = adjusted_revenue_indices[j]
                    if adjusted_revenue[index] <= max_revenue:
                        break
                    numerator += Decimal(utilities[t][index]) * adjusted_revenue[index]
                    denominator += Decimal(utilities[t][index])
                    revenue = numerator / denominator
                    if revenue > max_revenue:
                        max_revenue = revenue
                        final_index = j
                # Computing the next dynamic program value
                vee[t][x] = max_revenue + vee[t + 1][x]
                offer_sets[t][x] = adjusted_revenue_indices[:(final_index + 1)]
        self.offer_sets = offer_sets
        self.initial_inventory_tuple = initial_inventory_tuple
        self.num_products = num_products

    def check_offer_everything(self):
        """Checks if DP policy is the same as the offer everything policy, that is if for every inventory
         level and period the set it returns is all the available products."""
        non_oe_sets_count = 0
        for inventory_vector_to_set_dictionary in self.offer_sets:
            for inventory_vector, set in inventory_vector_to_set_dictionary.items():
                zero_count = 0
                # Count the number of products which have 0 inventory
                for x in inventory_vector:
                    if x == 0:
                        zero_count += 1
                # If the number of items in the set is equal to the number of products minus the products which
                # are not available (0 inventory) then the set must be offer everything.
                if len(set) != self.num_products - zero_count:
                    non_oe_sets_count += 1
        if non_oe_sets_count == 0:
            print("This policy is simply the OE policy.")
            return True
        else:
            print("This is not the OE policy")
            return False

    def offer_set(self, inventory, initial_inventory, t, customer):
        """Offers the DPO set, given a period and an initial inventory level

        Parameters
        ----------
        inventory:
            dictionary contains all in stock products as keys, and inventory values as values

        initial_inventory:
            dictionary which contains all products as keys, and initial inventory values as values
        t, customer:
            dummy parameter so that this method can be called by any policy object
        """
        # This class uses product keys instead of products for dictionary keys, so we have
        # to convert between the two.
        inventory_list = [0] * self.num_products
        product_key_to_product = {x.product_key: x for x in initial_inventory}
        for x in inventory:
            inventory_list[x.product_key] = inventory[x]
        inventory_tuple = tuple(inventory_list)
        return {product_key_to_product[x] for x in self.offer_sets[t][inventory_tuple]}

    def __str__(self):
        return "DPO"

    @staticmethod
    def customer_type_sensitive():
        return True

    @staticmethod
    def problem_instance_sensitive():
        return False


class OfferEverything:
    """Class implementing the offer everything policy"""

    def __init__(self, simulation):
        pass

    @staticmethod
    def offer_set(inventory, initial_inventory, t, customer):
        """Returns the assortment of all in stock products

        Parameters
        ----------
        inventory:
            dictionary which contains all in stock products as keys, and inventory values as values

        initial_inventory, t, customer:
            dummy parameter so that the method parameters are consistent with other policies
        """

        return inventory.keys()

    def __repr__(self):
        return "OE"

    def __str__(self):
        return "OE"

    @staticmethod
    def customer_type_sensitive():
        return False

    @staticmethod
    def problem_instance_sensitive():
        return False


def maximum_revenue_set(customer, adjusted_prices_sorted):
    """Returns the MNL revenue maximizing assortment

    Parameters
    ----------
    customer: customer
        object, which contains a list of attractions v_i for each product i in the problem instance, from
        which the MNL model can be derived
    adjusted_prices_sorted: list
        sorted in descending order by adjusted revenue. adjusted revenue is some metric attributed to each
        product, calculated by a policy which determines how valuable it is to sell any given product.

    Returns
    -------
    set:
     set of products which produces the greatest expected adjusted revenue by the MNL model
    """

    # Gallego and Topaloglu https://link.springer.com/book/10.1007/978-1-4939-9606-3 the set with the maximum
    # revenue has the form (1, ..., j), where (1,...,j,...n) are the products listed in descending order of
    # adjusted revenue. The MNL revenue of one of these sets of products (1, ..., j) is given by the expression
    # SUM_j(r_i w_i)/(1 + SUM_j(w_i)), where w_i is utility and r_i is adjusted revenue.
    # We compute this value for each of these sets by adding r_j w_j to r_1 w_1 + ... + r_(j-1)w_(j-1)
    # (the previous numerator) to calculate the numerator for each set and w_j to 1 + w_1 + ... + w_(j-1)
    # to calculate the next denominator.

    # expected revenue denominator. We use a convention in this program that the no purchase attraction is 1
    utility_sum = 1
    # expected revenue numerator
    offer_set_adjusted_revenue = 0
    max_revenue = 0
    final_index = -1
    for i in range(len(adjusted_prices_sorted)):
        product = adjusted_prices_sorted[i]
        # If the adjusted revenue of the next product is lower than the last expected revenue, it will "bring down"
        # the average, and every next product will have less than or equal to that amount of adjusted revenue,
        # so we know we have reached the optimal set and can break out of the loop
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
    ----------
    balancing_function: func
        function from [0,1]-->[0,1] which reduces the effective revenue of a product if its inventory is low
    """

    def __init__(self, policy_instance):
        def psi_eib(x):
            psi_c = np.exp(1) / (np.exp(1) - 1)
            return psi_c * (1 - np.exp(-x))

        self.balancing_function = psi_eib

    def offer_set(self, inventory, initial_inventory, t, arriving_customer_type_list):
        """
        Makes a product assortment decision, for which products to offer the customer arriving at period t

        Parameters
        ----------
        inventory: dictionary
            contains the products with non-zero inventory as keys and the current inventory levels at period t as values
        initial_inventory: dictionary
            contains all products as keys, and the initial inventory levels for the products as values
        t:
            index of the period in the selling horizon
        arriving_customer_type_list:
            The length of the list is the length of the selling horizon. Each index represents a period. The
            element at each index is a customer object of the customer type arriving during period.
        """
        # Adjusted price for each product i is p_i * phi(x_i/c_i), where phi is the inventory balancing function,
        # x_i is the inventory level, c_i is the initial inventory level, and p_i is the price of the product.
        # Reduces the likelihood that the policy will offer products with comparatively low inventory.
        adjusted_prices = dict((x, self.balancing_function(inventory.get(x) / initial_inventory.get(x)) * x.price)
                               for x in inventory)
        adjusted_prices_sorted = sorted(adjusted_prices.items(), key=lambda x: (-x[1], -x[0].product_key))
        return maximum_revenue_set(arriving_customer_type_list[t], adjusted_prices_sorted)

    def __repr__(self):
        return "IB"

    def __str__(self):
        return "IB"

    @staticmethod
    def customer_type_sensitive():
        return False

    @staticmethod
    def problem_instance_sensitive():
        return False


def generate_dpa_class_object(theta):
    """Function that returns a subclass of the DPAPolicy object with theta as an instance variable
    and problem instance as the only init method parameter"""

    class DPAReturnClass(DPAPolicy):
        def __init__(self, problem_instance):
            DPAPolicy.__init__(self, problem_instance, theta)

    return DPAReturnClass


class DPAPolicy:
    """Class implementing the DPA algorithm, comes from https://doi.org/10.1287/opre.2019.1931

    Attributes:

    gamma:
        list which stores gamma values for each product, at each period. Recall that gamma values are a crude approximation
        of the expected total revenue that can be extracted from a product at each period. Implemented as a list with
        indices being periods (0 indexed), and entries being dictionaries, with the products as keys and the
        expected total revenue as values.
    theta:
        tuning parameter used to calculate the gamma values. Adjusts the opportunity cost of losing inventory.
    """

    def __init__(self, problem_instance, theta):
        """
        Parameters
        ----------
        problem_instance: DynamicAssortmentOptimizationProblem
            problem_instance object, which contains all objects relevant to a single trial of a simulation.
            This policy requires the initial_inventory levels, and the arriving_customer_type_list objects
        theta: float
            tuning parameter used to calculate the gamma values. Adjusts the opportunity cost of losing inventory.
        """
        arriving_customer_type_list = problem_instance.arriving_customer_type_list
        initial_inventory = problem_instance.initial_inventory
        T = len(arriving_customer_type_list)

        # This is the gamma vector, which is calculated with a recursion.
        # Thus, we need one more period at the end of the selling horizon, to start the recursion
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
            adjusted_prices = dict((x, (x.price - theta * gamma[t + 1][x] / initial_inventory[x])) for x in
                                   initial_inventory)
            # Compute the "expected set" which we will use to calculate the purchase probabilities
            adjusted_prices_sorted = sorted(adjusted_prices.items(), key=lambda x: (-x[1], -x[0].product_key))
            max_set = maximum_revenue_set(arriving_customer_type_list[t], adjusted_prices_sorted)
            # Calculate the numerator for the probability calculation
            utility_sum = sum(arriving_customer_type_list[t].product_attractions[p] for p in max_set)
            for p in initial_inventory:
                if p in max_set:
                    # We split up the expression for gamma in two for readability
                    arg = p.price - theta * gamma[t + 1][p] / initial_inventory[p]
                    # This is the expression for the gamma recursion
                    gamma[t][p] = (arriving_customer_type_list[t].product_attractions[p] / utility_sum) * arg + \
                                  gamma[t + 1][p]
                else:
                    gamma[t][p] = gamma[t + 1][p]
        self.gamma = gamma
        self.coefficient = 1 / (1 - np.exp(-1))
        self.theta = theta

    def function(self, x):
        # Instead of re-computing self.coefficient every time this calculation is run,
        # we save it as an attribute. It saves a surprising amount of time since this function gets called so much
        return (1 - np.exp(-x)) * self.coefficient

    def basis_function(self, product, inventory, initial_inventory):
        """Function which tracks the availability of a product based on the ratio x between its
        inventory and its starting inventory."""
        return self.function(inventory.get(product) / initial_inventory.get(product))

    def offer_set(self, inventory, initial_inventory, t, arriving_customer_type_list):
        """Makes a product assortment decision, for which products to offer the customer arriving at period t

        Parameters
        ----------
        inventory:
            dictionary which contains the products with non-zero inventory as keys and the current inventory levels at period t as values
        initial_inventory:
            dictionary which contains all products as keys, and the initial inventory levels for the products as values
        t:
            index of the period in the selling horizon
        arriving_customer_type_list:
            list with length of the selling horizon, with a product for each element, where an element represents a period
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
        return maximum_revenue_set(arriving_customer_type_list[t], adjusted_prices_sorted)

    def __str__(self):
        return "DPA"

    @staticmethod
    def customer_type_sensitive():
        return True

    @staticmethod
    def problem_instance_sensitive():
        return False


class Clairvoyant:
    """ Class implementing the clairvoyant policy, which knows the decisions the customers will make in advance

    Attributes:

    offer_set: list
        with the same length as the selling horizon, and each element contains a set to offer the customers each period
    """

    def __init__(self, problem_instance):
        """
        Parameters
        ----------
        problem_instance: DynamicAssortmentOptimizationProblem
            problem_instance object, which contains all objects relevant to a single trial of a simulation.
            This class requires the product_utilities, no_purchase_utiltiies, and initial_inventory objects.
            In making use of the utility information, the policy knows ahead of time which decisions the customers
            will make.
        """
        product_utilities = problem_instance.product_utilities
        no_purchase_utilities = problem_instance.no_purchase_utilities
        initial_inventory = problem_instance.initial_inventory
        key_product_dict = {p.product_key: p for p in initial_inventory}
        T = len(product_utilities)
        num_products = len(initial_inventory)
        total_inventory = sum(initial_inventory.values())
        supplies = []

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
                supplies += [0]
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
            supplies += [0]
            i += 1

        # The minimum cost flow problem will be linked at it's sink and source nodes
        source_index = i
        supplies += [0]
        sink_index = i + 1
        supplies += [0]

        # All three of these lists will have the same length. For any index i, the rows list entry i
        # is the index of the source node of the edge, the column list entry contains the index of the
        # sink node, and the data list entry i contains the arc capacity for that edge.
        # The (negative) costs of the edges are the revenue of selling each product, and are also assigned to
        # each edge
        start_nodes = []
        end_nodes = []
        capacities = []
        unit_costs = []

        # generate the edges as described
        for t in range(T):
            for p in customers[t]:
                start_nodes += [t]
                end_nodes += [product_node_index[p]]
                capacities += [1]
                unit_costs += [-key_product_dict[p].price]

        for t in range(T):
            start_nodes += [source_index]
            end_nodes += [t]
            capacities += [1]
            unit_costs += [0]

        for p, x in initial_inventory.items():
            start_nodes += [product_node_index[p.product_key]]
            end_nodes += [sink_index]
            capacities += [x]
            unit_costs += [0]

        start_nodes += [sink_index]
        end_nodes += [source_index]
        capacities += [total_inventory]
        unit_costs += [0]

        # create and solve the matrix
        smcf = min_cost_flow.SimpleMinCostFlow()
        smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)
        for count, supply in enumerate(supplies):
            smcf.set_node_supply(count, supply)
        smcf.solve()

        # dictionary of edges
        graph_dictionary = dict()
        for i in range(smcf.num_arcs()):
            graph_dictionary[(smcf.tail(i), smcf.head(i))] = smcf.flow(i)

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

    @staticmethod
    def customer_type_sensitive():
        return False

    @staticmethod
    def problem_instance_sensitive():
        return True


def generate_topk_class_object(k):
    class TopKReturnClass(TopK):
        def __init__(self, problem_instance):
            TopK.__init__(self, problem_instance, k)

    return TopKReturnClass


class TopK:
    """Implements a policy which sells the k most expensive available products"""
    def __init__(self, problem_instance, k):
        self.k = k

    def offer_set(self, inventory, initial_inventory, t, sale_horizon):
        product_list = sorted({p for p in inventory.keys() if inventory[p] > 0}, key=lambda x: x.price)
        if self.k < len(product_list):

            return set(product_list[-self.k:])
        else:
            return set(product_list)

    def __str__(self):
        return "TopK " + str(self.k)

    @staticmethod
    def customer_type_sensitive():
        return False

    @staticmethod
    def problem_instance_sensitive():
        return False
