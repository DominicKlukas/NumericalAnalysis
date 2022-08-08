import numpy as np

class Customer:
    """A class used to represent a customer type

    Attributes
    __________
    products: dictionary
        products as keys and product purchase utilities as values
    no_purchase_utility: int
        utility of not buying any product
    customer_key: int
        an integer used to identify the product
    """

    def __init__(self, products, no_purchase_utility, customer_key):
        """
        Parameters
        __________
        products: dictionary
            products as keys and product purchase utilities as values
        no_purchase_utility: int
            utility of not buying any product
        customer_key: int
            an integer used to identify the product
        """
        self.products = products
        self.products_list = sorted(products.keys())
        self.no_purchase_utility = no_purchase_utility
        self.customer_key = customer_key

    def customer_decision(self, offered_set, customer_choice):
        if len(offered_set)==0:
            return None
        product_utilities = dict()
        i = 1
        for p in self.products_list:
            product_utilities[p] = np.log(self.products[p]) + customer_choice[i]
            i += 1
        np_utility = np.log(self.no_purchase_utility) + customer_choice[0]
        offered_set_utilities = {p : u for p, u in product_utilities.items() if p in offered_set}
        offered_set_utilities_list = sorted(offered_set_utilities.items(), key=lambda x: (-x[1], -x[0].product_key))
        chosen_product = offered_set_utilities_list[0][0]
        if np_utility > product_utilities[chosen_product]:
            return None
        else:
            return chosen_product

    def __lt__(self, other):
        return self.customer_key < other.customer_key

    def __repr__(self):
        return str(self.customer_key)
