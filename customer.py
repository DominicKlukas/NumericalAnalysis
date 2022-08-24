class Customer:
    """A class used to represent a customer type

    Attributes
    __________
    products:
        dictionary with products as keys and product attractions as values, relative to a no purchase utility of 1.
         Products attractions are defined as in https://link.springer.com/book/10.1007/978-1-4939-9606-3 chapter 4
    customer_key:
        an integer used to identify the product
    """

    def __init__(self, customer_key, product_attractions):
        """
        Parameters
        ----------
        product_attractions:
            dictionary with products as keys and product purchase attractions as values, relative to a no-purchase
             attraction of 1
        customer_key: int
            an integer used to identify the customer
        """
        self.product_attractions = product_attractions
        self.customer_key = customer_key

    def __lt__(self, other):
        return self.customer_key < other.customer_key

    def __repr__(self):
        return str(self.customer_key)
