import matplotlib.pyplot as plt
import numpy as np

def graph_plots(inventory_vectors, names, offered_sets, cumulative_revenue):
    figure, axis = plt.subplots(len(names)+1, 1)
    plt.subplots_adjust(hspace=0.8)
    figure.set_figwidth(12)
    figure.set_figheight(7)
    line_styles = ["solid", "dashed", "dotted"]
    colors = ['C3', 'C4', 'C2', 'C3', 'C4']
    for x in range(len(names)):
        axis[len(names)].plot(cumulative_revenue[x][:-1], label=names[x], linestyle=line_styles[x], color=colors[x],
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