from policy import *
from problem_instance import *


# Problem Setup
num_products = 5
C_i = [4, 4, 4, 4, 4]
np_utility = 1
b = 5
utilities = [b**(i-int(num_products/2)) for i in range(num_products)]
prices = [1,1,1, 1, 1]

T = 50

# In these tests, there are a couple things to do. First, we have to see how long it takes for all products
# to be sold out. Then, we increase the selling horizon until it is far enough past this that no matter what,
# products are always sold out at that point (at least, when the OE policy is used).

# simulate = Distribution3(T, num_products, np_utility, C_i, utilities, 0)
#
# policy = OfferEverything()
# output = simulate.simulation(policy)
# while sum(output[1][T-1]) != 0:
#     # Increase the period, hoping that the OE policy will sell out.
#     T += sum(output[1][T-1])/(min(utilities)/(min(utilities) + np_utility))
#     T = int(T)
#     simulate = Distribution3(T, num_products, np_utility, C_i, utilities, 0)
#     output = simulate.simulation(policy)

# Compute the optimal policy
p = TopalogluDPOptimal(C_i, prices, utilities, np_utility, T)
p.check_offer_everything(num_products)