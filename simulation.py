from problem_instance import *


def simulation(policy_generators, initial_inventory, prices, attractions, generate_arriving_customer_types, T, num_runs, seed):
    """ Function that runs simulations over multiple problem instances each initialized with the same parameters but
    with different random number generator seeds.

    Parameters
    ----------
    policy_generators:
        A list containing class objects. Each class object is a class implementing a policy, where the init
        method of the class has a problem_instance object as its only parameter.

    initial_inventory:
        list of initial inventory values of products, where the indices of the list match product key values

    prices:
        list of prices of products, where the indices of the list match product key values

    attractions:
        list of customer types, where each customer type is represented by a list of attraction values
        for each product. In a list of attraction values, the list indices correspond to product key values.
        IMPORTANT: No product attraction value can be zero, since this does not work in the MNL model.

    generate_arriving_customer_types:
        function which generates a list of length selling horizon (T) customer types, where those
        customer types are specified by the attractions list

    T: integer
        length of the selling horizon

    num_runs: integer
        number of problem instances to run, over which results are accumulated

    seed: number
        positive integer which is fed into the random number generators as seeds, to make simulations deterministic

    Returns
    -------
    tuple
        A tuple, containing the following elements. Each element is a list, where the elements in the list correspond
        to results from each of the policies. The results produced are as follows: Revenue: sum of the revenue earned
        by the policy over all the problem instances. inventory_vectors: For each policy, the element is a list, with
        one element per period in the selling horizon. The element corresponding to the period contains a list, which
        is the sum over all the trials of the inventory levels at that period. Offered_set_vector: For each policy,
        the element is a list where each element corresponds to a period. Each element corresponding to a period
        contains a list, where each element of the list corresponds to a product, and the value is the number of trials
        less the number of times the policy withheld that product during the period. cumulative_revenue: for each
        policy, the element is a list with elements for each period, where each element is the sum of the revenue
        earned up to that period on over all trials. names: name of the policy.
    """
    # generate the product objects (the same ones will be used for every simulation)
    product_list = []
    num_products = len(prices)
    for i in range(num_products):
        product_list += [Product(i, prices[i])]
    initial_inventory_dict = dict(zip(product_list, initial_inventory))

    # generate the customer objects (the same ones will be used for every simulation)
    customer_list = []
    for i in range(len(attractions)):
        product_attractions = dict(zip(product_list, attractions[i]))
        customer_list += [Customer(i, product_attractions)]

    num_policies = len(policy_generators)

    cumulative_output = []
    policies = []
    previous_arriving_customer_type = None
    customer_type_sensitive_policy_indices = []
    problem_instance_sensitive_policy_indices = []
    for i in range(num_runs):
        arriving_customer_type = generate_arriving_customer_types(customer_list, seed+i, T)
        simulate = DynamicAssortmentOptimizationProblem(product_list, initial_inventory_dict, arriving_customer_type,
                                                        seed + i)
        # Some policies need to do pre-calculation for specific problem instances, or only if the
        # sequence of customer type changes. We deal with this here.
        if i == 0:
            for k in range(num_policies):
                policies += [policy_generators[k](simulate)]
                if policies[k].customer_type_sensitive():
                    customer_type_sensitive_policy_indices += [k]
                if policies[k].problem_instance_sensitive():
                    problem_instance_sensitive_policy_indices += [k]
        else:
            if not check_customer_arrivals_same(previous_arriving_customer_type, arriving_customer_type):
                for k in customer_type_sensitive_policy_indices:
                    policies[k] = policy_generators[k](simulate)
            for k in problem_instance_sensitive_policy_indices:
                policies[k] = policy_generators[k](simulate)

        previous_arriving_customer_type = arriving_customer_type

        for j in range(num_policies):
            output = simulate.simulation(policies[j])
            if i == 0:
                # np.add can only be used with an array that is already initialized
                cumulative_output += [output]
            else:
                # We accumulate on the 4 outputs from the problem instance:
                # revenue, inventory vectors, offered set vector, and cumulative revenue
                # from the current run to all of the previous runs
                cumulative_output[j] = [np.add(cumulative_output[j][k],output[k]) for k in range(4)]

    revenue = [cumulative_output[i][0] for i in range(num_policies)]
    inventory_vectors = [cumulative_output[i][1] for i in range(num_policies)]
    offered_sets = [cumulative_output[i][2] for i in range(num_policies)]
    cumulative_revenue = [cumulative_output[i][3] for i in range(num_policies)]
    names = [str(policies[i]) for i in range(num_policies)]
    return revenue, inventory_vectors, offered_sets, cumulative_revenue, names


def check_customer_arrivals_same(previous_arriving_customer_type, arriving_customer_type):
    """Checks if two customer arrival sequences are the same
    Parameters
    ----------
    previous_arriving_customer_type:
        first sequence of customer types to compare
    arriving_customer_type:
        second sequence of customer types to compare
    """
    same_customer_types = True
    for x in range(len(previous_arriving_customer_type)):
        if previous_arriving_customer_type[x] != arriving_customer_type[x]:
            same_customer_types = False
            break
    return same_customer_types
