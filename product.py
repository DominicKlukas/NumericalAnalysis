class Product:
    """A class used to represent a product type

    Attributes
    __________
    product_key: int
        an integer used to identify the product
    price: int
        price of the product, equal to revenue earned when sold
    """

    def __init__(self, product_key, price):
        """
        Parameters
        __________
        product_key: int
            an integer used to identify the product
        price: int
            Price of the product, equal to revenue earned when sold
        """
        self.product_key = product_key
        self.price = price

    def __lt__(self, other):
        """Return true if self's key is smaller in the strict ordering than other's"""
        return self.product_key < other.product_key

    def __str__(self):
        return str(self.product_key)

    def __repr__(self):
        return self.__str__()
