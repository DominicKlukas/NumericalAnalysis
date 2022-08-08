import numpy as np
from product import Product
from customer import Customer
import matplotlib.pyplot as plt

def dirichlet_list(dimension, variance, length):
    alpha = (variance,) * dimension
    c = np.random.dirichlet(alpha, 1)[0]
    c *= length
    c = np.round(c).astype(int)
    # As a result of the rounding, the parameters may no longer sum to length
    while np.sum(c) > length:
        c[np.argmax(c)] -= 1
    while np.sum(c) < length:
        c[np.argmin(c)] += 1
    z = []
    y = 0
    for x in c:
        z += [y] * x
        y += 1
    z = np.random.permutation(z)
    return z.astype(int)


def plot_customer_distributions(customer_set, name):
    num_customers = len(customer_set)
    customer_list = sorted(customer_set)
    i = 1
    j = 1
    while i * j < num_customers:
        if i * j >= (i + 1) * (i + 1):
            j = i + 2
            i += 1
        else:
            j += 1

    figure, axis = plt.subplots(i, j)
    plt.subplots_adjust(hspace=0.4, wspace=0.5)
    bars = []
    for c in customer_list:
        products = c.products.keys()
        products_list = sorted(products)
        weights = [c.no_purchase_utility]
        for p in products_list:
            weights += [c.products[p]]
        normalize = sum(weights)
        for x in range(len(weights)):
            weights[x] = weights[x] / normalize
        bars += [weights]

    figure.set_figwidth(j*3)
    figure.set_figheight(i*3.5)

    bar_collection = []
    if i > 1:
        for x in range(num_customers):
            bar_collection += [axis[int(x / j), x % j].bar(range(len(bars[x])), bars[x])]
            axis[int(x / j), x % j].set_title("Customer " + str(x+1), fontsize=12)
            axis[int(x / j), x % j].set_xlabel("Product Index", fontsize=8)
            axis[int(x / j), x % j].set_ylabel("Utility", fontsize=8)
    elif j > 1:
        for x in range(num_customers):
            bar_collection += [axis[x].bar(range(len(bars[x])), bars[x])]
            axis[x].set_title("Customer " + str(x+1), fontsize=12)
            axis[x].set_xlabel("Product Index", fontsize=8)
            axis[x].set_ylabel("Utility", fontsize=8)
    else:
        for x in range(num_customers):
            bar_collection += [axis.bar(range(len(bars[x])), bars[x])]
            axis.set_title("Customer " + str(x+1), fontsize=12)
            axis.set_xlabel("Product Index", fontsize=8)
            axis.set_ylabel("Utility", fontsize=8)
    for x in range(num_customers):
        bar_collection[x][0].set_color('r')
    figure.suptitle(name + " (red bar = no purchase utility)")# "Customer Distributions for " +
    plt.plot()
    plt.show()


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
        probability_vectors = []
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
            probability_sum = sum(self.sale_horizon[t].products.values()) + self.sale_horizon[t].no_purchase_utility#[p] if p in inventory else 0 for p in offered_set) + self.sale_horizon[t].no_purchase_utility
            probability_vectors += [
                [np.round(100*self.sale_horizon[t].products[key_product_dict[i]] / probability_sum)/100 for i in range(num_products)]]

            if product_chosen is not None:
                revenue += product_chosen.price
                inventory.update({product_chosen: inventory.get(product_chosen) - 1})
                if inventory.get(product_chosen) == 0:  # This makes the code faster.
                   inventory.pop(product_chosen)

            cumulative_revenue += [revenue]

        # おわり
        return revenue, inventory_vectors, offered_set_vector, cumulative_revenue, probability_vectors


class SingleCustomerType(DynamicAssortmentOptimizationProblem):
    """Single Customer Type"""
    def __init__(self, T, num_products, np_utility, C_i, utilities, prices, seed):
        products = set()
        for i in range(num_products):
            products.add(Product(i, prices[i]))
        product_list = sorted(products)
        initial_inventory = dict()
        k = 0
        for x in product_list:
            initial_inventory[x] = C_i[k]
            k += 1

        product_utility_dict = dict(zip(product_list, utilities))
        customer = Customer(product_utility_dict, np_utility, 0)
        sale_horizon = [customer] * T

        np.random.seed(seed)
        customer_decision_seed = np.random.uniform(0, 1, T)
        self.C_i = C_i
        self.T = T
        self.num_products = num_products
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
        dirlist = dirichlet_list(num_products, (1 / num_products) * ((num_products - 1) / C_v ** 2 - 1), T)
        customer_list = sorted(customers)
        sale_horizon = [customer_list[x] for x in dirlist]

        customer_decision_seed = np.random.uniform(0, 1, T)
        self.C_i = C_i
        self.T = T
        self.num_products = num_products
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