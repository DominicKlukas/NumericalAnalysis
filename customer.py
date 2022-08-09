import numpy as np

class Customer:
    """A class used to represent a customer type

    Attributes
    __________
    products: dictionary
        products as keys and product purchase utilities as values, relative to a no purchase utility of 1
    customer_key: int
        an integer used to identify the product
    """

    def __init__(self, customer_key, product_attractions):
        """
        Parameters
        __________
        products: dictionary
            products as keys and product purchase utilities as values, relative to a no purchase utility of 1
        customer_key: int
            an integer used to identify the product
        """
        self.product_attractions = product_attractions
        self.customer_key = customer_key

    def __lt__(self, other):
        return self.customer_key < other.customer_key

    def __repr__(self):
        return str(self.customer_key)
