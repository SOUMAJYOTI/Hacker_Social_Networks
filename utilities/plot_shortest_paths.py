import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.patches as patches
from pylab import *
import pickle


def probBoxPlots(data, t):
    '''
    dict: key - month; values = list of time differences
    :param dict:
    :return:
    '''

    # titles = []
    # data_to_plot = []
    #
    # for d in range(len(data)):
    #     titles.append(d)
    #     data_to_plot.append(v)

    # Create the box_plots
    fig = plt.figure(1, figsize=(10, 8))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data, patch_artist=True)

    for box in bp['boxes']:
        # change outline color
        box.set(color='#0000FF', linewidth=2)
        # change fill color
        box.set(facecolor='#FFFFFF')

        ## change color and linewidth of the whiskers
        # for whisker in bp['whiskers']:
        #     whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        # for cap in bp['caps']:
        #     cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#FF0000', linewidth=4)

        ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # ax.set_title('Motif transition:' + str(m4) + '-->' + str(m5))
    # ax.set_ylabel('Shortest Path', size=)
    # ax.set_ylim([0, 0.05])
    ax.set_xticklabels(titles, rotation=60, ha='right')

    third_quartile = [item.get_ydata()[0] for item in bp['whiskers']]
    third_quartile = max(third_quartile)

    # dir_save = '../plots/motif_transition_plots/12_08/v2/inhib'
    # if not os.path.exists(dir_save):
    #     os.makedirs(dir_save)
    # file_save = dir_save + '/' + 'mt_inhib_' + str(m4) + '_' + str(m5) + '.png'

    plt.tick_params('y', labelsize=20)
    plt.tick_params('x', labelsize=20)
    plt.xlabel('Date Timeframe (Start of each week)', fontsize=25)
    # plt.ylabel('Days', fontsize=25)
    plt.title('Shortest Path between two groups experts and training users', fontsize=25)
    plt.subplots_adjust(left=0.12, bottom=0.25, top=0.95)

    # if m4 not in limits_y_steep:
    #     limits_y_steep[m4] = {}
    # try:
    #     limits_y_steep[m4][m5] = third_quartile + math.pow(10, int(math.log10(third_quartile)))
    # except:
    #     limits_y_steep[m4][m5] = 0

    plt.grid(True)
    plt.show()
    # plt.savefig(file_save)
    plt.close()


if __name__ == "__main__":
    y1 = pickle.load(open('../data/DW_data/09_15/train/features/spath_allExpertsKB_alltrainUsers.pickle', 'rb'))
    y2 = pickle.load(open('../data/DW_data/09_15/train/features/spath_ExpertsOverMeanKB_alltrainUsers.pickle', 'rb'))
    titles = pickle.load(open('../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))

    probBoxPlots(y1, titles)