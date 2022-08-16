import matplotlib.pyplot as plt
import numpy as np


def graph_inventory_plots(inventory_vectors, names, offered_sets, cumulative_revenue):
    figure, axis = plt.subplots(len(names)+1, 1)
    plt.subplots_adjust(hspace=0.8)
    figure.set_figwidth(12)
    figure.set_figheight(7)
    line_styles = ["solid", "dashed", "dotted"]
    for x in range(len(names)):
        axis[len(names)].plot(cumulative_revenue[x][:-1], label=names[x], linestyle=line_styles[x],
                              linewidth=2.5)
    axis[len(names)].legend(prop={'size': 8})
    axis[len(names)].set_title('Revenue vs Period', size=15)
    axis[len(names)].set_xlabel('Period', size=12)
    axis[len(names)].set_ylabel("Revenue", size=12)
    axis[len(names)].set_xticks(range(len(inventory_vectors[0])), np.arange(1, len(inventory_vectors[0]) + 1, 1))

    for policy in range(len(names)):
        inventory_vector = inventory_vectors[policy]
        num_products = len(inventory_vector[0])
        product_inventory_vectors = []
        ec = []
        for x in range(num_products):
            product_inventory_vectors += [[inventory_vector[t][x] for t in range(len(inventory_vector))]]
            ec += [['None' if offered_sets[policy][t][x] == 1 else "red" for t in range(len(inventory_vector))]]

        X_axis = np.arange(len(inventory_vector))

        gap = (1-0.5)/num_products
        start = gap*num_products/2-gap/2

        for x in range(num_products):
            axis[policy].bar(X_axis + gap*x - start, product_inventory_vectors[x], gap,
                             color=((1/num_products)*x, (1/num_products)*x, (1/num_products)*x), label="Product " + str(x+1), edgecolor=ec[x])
        axis[policy].set_ylabel("Inventory", size=12)
        axis[policy].set_xticks(range(len(inventory_vector)), np.arange(1, len(inventory_vector)+1, 1))
        axis[policy].set_title("Inventory Levels over Time for " + names[policy], size=15)
    axis[0].legend(prop={'size': 8})
    plt.show()


def plot_cumulative_revenue(policy_names, cumulative_revenue,num_runs):
    max_value = np.max([np.argmax(cumulative_revenue[x]) for x in range(len(policy_names))])
    fig = plt.figure()
    for i in range(len(policy_names)):
        function = [cumulative_revenue[i][x]/num_runs for x in range(len(cumulative_revenue[i]))]
        plt.plot(function, label=policy_names[i])
    plt.title("Average Revenue vs Period", pad=20, size=25)
    plt.xlabel("Period", size=15)
    plt.ylabel("Average Revenue", size=15)
    plt.legend(prop={'size': 12})
    plt.grid()
    fig.set_figwidth(15)
    fig.set_figheight(7.25)
    plt.show()


def plot_finite_difference(cumulative_revenue, num_runs, names):
    differences = []
    fig = plt.figure()
    for x in range(len(names)):
        differences += [
            [(cumulative_revenue[x][t + 1] - cumulative_revenue[x][t])/num_runs for t in range(len(cumulative_revenue[x]) - 1)]]
        plt.plot(np.arange(1, len(differences[x]) + 1, 1), differences[x], label=names[x])
    plt.legend(prop={'size':12})
    plt.grid()
    plt.ylabel("Average Revenue Per Period", size=15)
    plt.xlabel("Period", size=15)
    plt.xticks([1, ] + list(np.arange(5, len(differences[0]) + 2, 5)))
    fig.set_figwidth(15)
    fig.set_figheight(7.25)
    plt.show()


def plot_revenue_vs(revenue_vs_parameter, policy_names, parameter_name, num_runs, tick_size):
    """
    Graphs the performance of the policies relative to another policy
    Parameters
    __________
    revenue_vs_parameter: dict
        with each tested parameter value as keys, and list containing the revenue for each policy at that
        tested parameter value as a dictionary value
    policy_names: list
        of policy names, with the indexing corresponding to the indexing of the revenue lists
    parameter_name: string
        name of the parameter being tested, for display on the graph
    """
    domain = sorted(revenue_vs_parameter.keys())
    linestyles = ['solid', 'solid', 'dashed', 'solid', 'dotted']
    for i in range(len(policy_names)):
        function = [revenue_vs_parameter[x][i] / num_runs for x in domain]
        plt.plot(domain, function, label=policy_names[i], linestyle = linestyles[i])
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, max(domain)+tick_size, tick_size))
    # plt.xticks([0, 4, 8], ['(1, 3, 5, 7, 9)', '(5, 5, 5, 5, 5)', '(9, 7, 5, 3, 1)'])
    plt.xlabel(parameter_name, size=12.5)
    plt.ylabel("Average Revenue", size=12.5)
    plt.axvline(x = 25, color = 'black')

#    plt.title(parameter_name + " vs Average Revenue", size=15)
    plt.show()


def plot_ratio_optimal(revenue_vs_parameter, policy_names, parameter_name, baseline_policy_index, tick_size):
    """
    Graphs the performance of the policies relative to another policy
    Parameters
    __________
    revenue_vs_parameter: dict
        with each tested parameter value as keys, and list containing the revenue for each policy at that
        tested parameter as a value
    policy_names: list
        list of policy names, with the indexing corresponding to the indexing of the revenue lists
    parameter_name: string
        name of the parameter being tested, for display on the graph
    baseline_policy_index: integer
        index of the policy that the other policies are being compared to
    """
    colors = ['C1', 'C2', 'C3', 'C4', 'C5']
    domain = sorted(revenue_vs_parameter.keys())
    for i in range(len(policy_names)-1):
        function = [revenue_vs_parameter[x][i+1]/revenue_vs_parameter[x][baseline_policy_index] for x in domain]
        plt.plot(domain, function, label=policy_names[i+1], color=colors[i])
    plt.grid()
    plt.legend()
    plt.axvline(x = 25, color='black')
    plt.xticks(np.arange(0, max(domain)+tick_size, tick_size))
    plt.xlabel(parameter_name, size=12.5)
    plt.xlabel(parameter_name, size=12.5)
    plt.ylabel("Performance Ratio", size=12.5)
    plt.show()
