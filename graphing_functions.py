import matplotlib.pyplot as plt
import numpy as np
from simulation import *
import pandas as pd
import openpyxl


def graph_inventory_plots(inventory_vectors, names, offered_sets, cumulative_revenue):
    """Graphs the inventory levels over time for different policies, for a single run. Ensure that the results
    from the parameters are for a single run."""
    figure, axis = plt.subplots(len(names)+1, 1)
    plt.subplots_adjust(hspace=0.8)
    figure.set_figwidth(12)
    figure.set_figheight(7)
    line_styles = ["solid", "dashed"] + ["dotted"]*len(names)
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
    """Graphs the cumulative revenue of the different policies over the selling horizon"""
    max_value = np.max([np.argmax(cumulative_revenue[x]) for x in range(len(policy_names))])
    fig = plt.figure()
    for i in range(len(policy_names)):
        function = [cumulative_revenue[i][x]/num_runs for x in range(len(cumulative_revenue[i]))]
        plt.plot(function, label=policy_names[i])
    plt.xlabel("Period", size=15)
    plt.ylabel("Average Revenue", size=15)
    plt.legend(prop={'size': 12})
    plt.grid()
    fig.set_figwidth(15)
    fig.set_figheight(7.25)
    plt.show()


def plot_revenue_per_period(cumulative_revenue, num_runs, names):
    """Plots the revenue earned per period at each point in the selling horizon"""
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


def plot_revenue_vs_parameter(revenue_vs_parameter, policy_names, parameter_name, num_runs, tick_size):
    """
    Graphs the performance of the policies relative to another policy

    Parameters
    ----------
    revenue_vs_parameter: dict
        with each tested parameter value as keys, and list containing the revenue for each policy at that
        tested parameter value as a dictionary value
    policy_names: list
        of policy names, with the indexing corresponding to the indexing of the revenue lists
    parameter_name: string
        name of the parameter being tested, for display on the graph
    num_runs:
        the number of trials the results have been cumulated over, so that the average can be computed
    tick_size:
        the distance between the labels on the x-axis
    """
    domain = sorted(revenue_vs_parameter.keys())
    linestyles = ['solid', 'solid', 'dashed', 'solid', 'dotted']
    for i in range(len(policy_names)):
        function = [revenue_vs_parameter[x][i] / num_runs for x in domain]
        plt.plot(domain, function, label=policy_names[i], linestyle = linestyles[i])
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(min(domain), max(domain)+tick_size, tick_size))
    plt.xlabel(parameter_name, size=12.5)
    plt.ylabel("Average Revenue", size=12.5)

#    plt.title(parameter_name + " vs Average Revenue", size=15)
    plt.show()


def plot_ratio_optimal_vs_parameter(revenue_vs_parameter, policy_names, parameter_name, baseline_policy_index,
                                    tick_size):
    """
    Graphs the performance of the policies relative to another policy

    Parameters
    ----------
    revenue_vs_parameter: dict
        with each tested parameter value as keys, and list containing the revenue for each policy at that
        tested parameter as a value
    policy_names: list
        list of policy names, with the indexing corresponding to the indexing of the revenue lists
    parameter_name: string
        name of the parameter being tested, for display on the graph
    baseline_policy_index: integer
        index of the policy that the other policies are being compared to
    tick_size:
        the distance between the labels on the x-axis
    """
    colors = ['C1', 'C2', 'C3', 'C4', 'C5']
    domain = sorted(revenue_vs_parameter.keys())
    for i in range(len(policy_names)):
        function = [revenue_vs_parameter[x][i]/revenue_vs_parameter[x][baseline_policy_index] for x in domain]
        plt.plot(domain, function, label=policy_names[i], color=colors[i])
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(min(domain), max(domain)+tick_size, tick_size))
    plt.xlabel(parameter_name, size=12.5)
    plt.xlabel(parameter_name, size=12.5)
    plt.ylabel("Performance Ratio", size=12.5)
    plt.show()


def create_excel_table(policies, prices_list, initial_inventory_list, attractions_list, sale_horizon_length_list,
                       num_runs, customer_type_generating_function, baseline_policy_index, filepath, seed):
    """
    Function that generates an exel table of results, for a set of experiments. Each row in the exel table
    lists a set of parameters and the results for those parameters. The spreadsheet displays the results for
    every combination of parameters provided in the arguments. The exel table has a column for each
    policy being tested, and reports the average revenue per trial for one baseline policy and the ratio of the
    average revenue between each other policy and the baseline average revenue for all the other policies.

    Args:
        policies:
            A list of class objects, where each class is a policy implementation. The class gets
            instantiated in simulation.py, where it receives the required randomly generated customers
            and parameters for each trial
        prices_list:
            List of lists, where each list is the list of prices for the products.
        initial_inventory_list:
            List of lists, where each list is the initial inventory levels for the products.
        attractions_list:
            A list of a list of a list. We permute through each list of lists. Each list of lists represents
            a set of customer types, where each customer type is given by a list of attraction values attributed
            to each product.
        sale_horizon_length_list:
            function of the initial inventory, which generates a list of selling horizon lengths to permute through.
            The selling horizon lengths must be integers.
        num_runs:
            number of trials for each experiment
        customer_type_generating_function:
            function which determines how the different customer types defined in each element of attractions_list are
            distributed over the selling horizon
        baseline_policy_index:
            index of the policy in policies which the other policies should be compared to
        filepath:
            pathname of the file to output
        seed:
            Integer which uniquely determines the outcome of the simulation

    Returns:
        Generates an exel spreadsheet with results for each combination of parameters
    """
    data = [['Prices', 'Attractions', 'Initial Inventory', 'T', 'Baseline Policy'] + ['' for x in range(len(policies))]]
    for prices in prices_list:
        for attractions in attractions_list:
            for inventory in initial_inventory_list:
                for T in sale_horizon_length_list(inventory):
                    revenue, inventory_vectors, offered_sets, cumulative_revenue, names = \
                        simulation(policies, inventory, prices, attractions, customer_type_generating_function, T, num_runs, seed)
                    data[0][-len(policies):] = names
                    data += [[tuple(prices), tuple(attractions[0]), tuple(inventory), T, revenue[0] / num_runs]
                             + [revenue[n] / revenue[baseline_policy_index] for n in range(len(policies))]]
    # convert your array into a dataframe
    df = pd.DataFrame(data)
    # save to xlsx file
    df.to_excel(filepath, index=False, header=False)


def plot_box_graphs(clairvoyant_revenues, offer_everything_revenues, x_labels):
    """Plots box charts comparing the clairvoyant's performance to the offer everything policy, for different experiments

    Parameters
    ----------
    clairvoyant_revenues:
        List with one entry per experiment. Each experiment has a list of revenues for different trials of the
        same experiment
    offer_everything_revenues:
        List with one entry per experiment. Each experiment has a list of revenues for different trials of the
         same experiment. Must be same dimensino as clairvoyant_revenues
    x_labels:
        labels for each of the box plots

    Returns
    -------
    graph
        Box plot, displaying the distribution data of the ratio between the clairvoyant and offer everything revenue
        at each trial.
    """
    latex_font = {'fontname': 'cmr10'}
    data = [[clairvoyant_revenues[x][y]/offer_everything_revenues[x][y] if offer_everything_revenues[x][y] != 0 else 1 for
             y in range(len(clairvoyant_revenues[x])) ] for x in range(len(clairvoyant_revenues))]
    ax = plt.axes()
    plt.subplots_adjust(bottom=0.15, left=0.185)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # increase tick width
    ax.tick_params(width=2)
    plt.xticks(size = 18, **latex_font)
    plt.yticks(size = 18, **latex_font)
    boxplot = plt.boxplot(data,labels=x_labels,flierprops={'markersize': 2, 'markerfacecolor': 'gray', 'linewidth':0, 'markeredgecolor':'gray'}, whiskerprops={'linewidth':2},
                          medianprops={'linewidth':2,'linestyle':'--'})
    caps = boxplot['caps']
    for c in caps:
        c.set_linewidth(2)
    boxes = boxplot['boxes']
    for b in boxes:
        b.set_linewidth(2)
    plt.xlabel("Initial Inventory", size=25, **latex_font)
    plt.ylabel("CLV/OE Revenue", size=25, **latex_font)
    plt.show()


def plot_comparison_charts(baseline_revenue, x_labels, policy_revenues, policy_names, ylabel, colors, patterns):
    latex_font = {'fontname':'cmr10'}
    X_axis = np.arange(len(baseline_revenue))
    plt.subplots_adjust(bottom=0.15, left=0.185)
    ax = plt.axes()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    # increase tick width
    ax.tick_params(width=2)
    gap = 0.25
    start = ((len(policy_revenues[0])-1)/2)*gap
    thickness = 0.2
    policy_ratios = []
    for x in range(len(policy_revenues[0])):
        policy_ratios += [[policy_revenues[y][x]/baseline_revenue[y] for y in range(len(baseline_revenue))]]
    max_y_lim = int(max(max(policy_ratios))*100)/100+0.02
    min_y_lim = int(min(min(policy_ratios))*100)/100-0.02
    for x in range(len(policy_revenues[0])):
        plt.bar(X_axis + gap*x - start, policy_ratios[x], thickness, label=policy_names[x], color=colors[x], hatch=patterns[x])
    plt.xlabel("Initial Inventory", size = 25, **latex_font)
    plt.ylabel(ylabel, size = 25, **latex_font)
    plt.axhline(y = 1, color="gray")
    plt.xticks(X_axis, x_labels, size=18, **latex_font)
    plt.ylim(min_y_lim, max_y_lim)
    plt.yticks(size=18, **latex_font)
    plt.legend()
    plt.show()
