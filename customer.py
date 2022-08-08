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
        self.no_purchase_utility = no_purchase_utility
        self.customer_key = customer_key

    def customer_decision(self, offered_set, customer_choice):
        """Returns the customer's choice, from an offered set of products

        Parameters
        __________
        offered_set: set
            product objects, from which the customer is to purchase a product or choose not to purchase a product
        customer_choice: float
            float between 0 and 1 which uniquely determines the customer's choice, given offered_set

        Returns
        _______
        product
            Either None, or a product in offered_set which the customer has chosen to purchase
        """
        # MNL probability is given by w_i/(w_(np) + sum([w])). Compute this denominator.
        bam_den = sum(self.products.get(x) for x in offered_set) + self.no_purchase_utility
        # offered_list will have a unique ordering, since every product has a unique index
        offered_list = sorted(offered_set)
        # For each product i, 0 <= (w_(np) + w_1 + ... + w_i)(w_(np) + sum([w])) <= 1.
        # The customer chooses the smallest product i such that
        # customer_choice < (w_(np) + w_1 + ... + w_i)(w_(np) + sum([w])), which could also be
        # the no purchase option, w_(np)
        # Instead of normalizing the utilities, we simply compute
        # customer_choice*(w_(np) + sum([w])) < (w_(np) + w_1 + ... + w_i) instead.
        acc = self.no_purchase_utility

        customer_choice *= bam_den
        i = -1
        while acc < customer_choice:
            i += 1
            acc += self.products.get(offered_list[i])
        if i == -1:
            return None
        else:
            return offered_list[i]

    def __lt__(self, other):
        return self.customer_key < other.customer_key

    def __repr__(self):
        return str(self.customer_key)
