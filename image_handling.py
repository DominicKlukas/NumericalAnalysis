import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import numpy as np

import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'/Users/dominic/bin/JDownloader 2.0/tools/mac/ffmpeg_10.10+/ffmpeg'

def colour_gradient(first_colour, second_colour, gradient):
    first = ((1-gradient)*first_colour[0] + gradient*second_colour[0])/255
    second = ((1 - gradient)*first_colour[1] + gradient*second_colour[1])/255
    third = ((1 - gradient)*first_colour[2] + gradient*second_colour[2])/255
    return (first, second, third)

def create_gif(policy_names, inventory_vectors, offered_sets, cumulative_revenue,
               probability_vector, animation_length, num_runs):

    sale_horizon = np.max([np.argmax(cumulative_revenue[x]) for x in range(len(policy_names))])

    num_products = len(inventory_vectors[0][0])
    titles = []
    for i in range(len(policy_names)):
        titles += [str(policy_names[i])]

    figure, axis = plt.subplots(2, 3)
    plt.subplots_adjust(hspace=0.4, bottom = 0.2)
    figure.set_figwidth(15)
    figure.set_figheight(7)
    figure.suptitle("Animation")

    indices = int(sale_horizon/(30*animation_length)) #5 seconds at 30 frames per second
    if indices == 0:
        indices = 1

    support = range(0, sale_horizon, indices)

    product_indices = [j + 1 for j in range(num_products)]
    barcollection = []
    text_collection = dict()
    probability_text = []

    patterns = [2.5, 2.5, 1.5, 1.5, 1.5]
    colors = ['b', 'green', 'orange', 'red', 'mediumpurple']

    rows = ["Ranking", "Policy", "Cumulative Revenue"]
    cols = [[1, 2, 3, 4, 5],["" for c in range(5)],["" for c in range(5)]]

    table = axis[1, 1].table(
        cellText=cols,
        rowLabels=rows,
        cellLoc='left',
        loc='upper left',
        bbox = [-0.7, -0.65, 3, 0.4])

    zoomed = figure.add_axes([0.8, 0.29, 0.1, 0.1])

    for i in range(len(policy_names)):
        axis[1,2].plot(range(sale_horizon), cumulative_revenue[i][:sale_horizon],
                       linewidth=patterns[i], label=policy_names[i], color=colors[i])
    axis[1,2].set_title("Revenue vs Period")
    axis[1,2].set_xlabel("Period", fontsize=8)
    axis[1,2].set_ylabel("Revenue", fontsize=8)
    axis[1,2].legend(loc=4, prop={'size':5})
    axis[1,2].grid()
    y_max = cumulative_revenue[0][len(cumulative_revenue[0])-1]
    position_line, = axis[1,2].plot([0, 0], [0, y_max], color='black')

    for i in range(len(policy_names)):
        axis[int(i/3), i%3].set_title(titles[i], fontsize=12)
        axis[int(i/3), i%3].set_xlabel("Product Index", fontsize=8)
        axis[int(i/3), i%3].set_ylabel("Inventory Level", fontsize=8)
        axis[int(i / 3), i % 3].set_ylim(0, max(inventory_vectors[i][0]) + 1)
        probability_text += [[axis[1, 1].text(0.6 * i - 0.68, -0.47, "", fontsize=8, transform=axis[1, 1].transAxes),
                              axis[1, 1].text(0.6 * i - 0.68, -0.61, "", fontsize=10, transform=axis[1, 1].transAxes)]]#,
                              # axis[1, 1].text(0.6 * i - 0.68, -0.70, "", fontsize=10, transform=axis[1, 1].transAxes)]]
        barcollection += [axis[int(i/3), i%3].bar(product_indices, inventory_vectors[i][0])]
        for x in range(num_products):
            b = barcollection[i][x]
            text_collection[b] = axis[int(i/3), i%3].text(b.get_xy()[0]+b.get_width()/2, b.get_height() + 0.5, str(0))

    max_difference = 0


    print(text_collection)


    def plot_function(frames):
        zoomed.clear()
        period = support[frames]
        indices_array = list(np.argsort([-cumulative_revenue[x][period] for x in range(len(policy_names))]))
        start = max(0, period - 50)

        position_line.set_data([frames*indices, frames*indices],[0, y_max])
        for j in range(len(policy_names)):
            zoomed.plot(cumulative_revenue[j][start:period], linewidth=patterns[j],
                        label=policy_names[j], color=colors[j])
            probability_text[indices_array.index(j)][0].set_text(policy_names[j])
            probability_text[indices_array.index(j)][1].set_text(str(cumulative_revenue[j][period]))
            # probability_text[indices_array.index(j)][2].set_text(str(cumulative_probabilities[j][period]))
            for i in range(num_products):
                barcollection[j][i].set_height(inventory_vectors[j][period][i])
                color = offered_sets[j][period][i]/num_runs
                barcollection[j][i].set_color(colour_gradient((8, 58, 8),(126, 126, 126), color))
                b = barcollection[j][i]
                text_collection[b].set_position((b.get_xy()[0] + b.get_width() / 2 - 0.05, b.get_height() + 0.5))
                text_collection[b].set_text(np.round(probability_vector[j][period][i]*100)/100)
        zoomed.set_xlim(30, 60)
        zoomed_y_max = cumulative_revenue[indices_array[0]][period]
        zoomed_y_min = cumulative_revenue[indices_array[len(policy_names)-1]][period]
        zoomed.set_ylim(zoomed_y_min-10-(zoomed_y_max - zoomed_y_min)*0.5,zoomed_y_max + 10+(zoomed_y_max - zoomed_y_min)*0.5)
        zoomed.xaxis.set_visible(False)
        zoomed.yaxis.set_visible(False)

    anim = FuncAnimation(figure, plot_function, frames=len(support), interval=30, blit = False)
    plt.show()
