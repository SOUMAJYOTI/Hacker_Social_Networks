import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.patches as patches
from pylab import *
import pickle

def plot_line(x, y1, y2, l1, l2, xLabels=''):
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111)  # row x col x position (here 1 x 1 x 1)

    plt.xticks(range(len(xLabels)), xLabels, size=30, rotation=60, ha='right')  # rotate x-axis labels to 75 degree
    plt.yticks(size=30)

    x = np.array(range(len(x)))
    # print(x.shape, len(y1))
    ax.plot(x, y1,  marker='o', linestyle='-', label=l1, linewidth=3)
    ax.plot(x, y2, marker='o', linestyle='-', label=l2, linewidth=3)
    # ax.plot(x, y3, marker='o', linestyle='-', label=l3, linewidth=3)
    # ax.plot(x, y4, marker='o', linestyle='-', label=l4, linewidth=3)
    # ax.plot(x, y5, marker='o', linestyle='-', label=l5, linewidth=3)

    # plt.xlim(0, len(var) + 1)
    plt.tight_layout()  # showing xticks (as xticks have long names)
    ax.grid()

    # plt.title(title, color='#000000', weight="bold", size=30)
    plt.ylabel('Mean Absolute Error', size=40)
    plt.xlabel('Interval ranges before $N_{inhib}$', size=40)

    ax.legend(loc='upper right', fancybox=True, shadow=True, fontsize=25)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.90)

    # plt.ylim([0, 1])
    # plt.xlim([0, 6])
    # plt.savefig('outputs/v4/edges_reg_motifs/Motif_' + m + '.png')
    # plt.savefig('outputs/v4/edges_reg_motifs/Motif_' + m + '.pdf')
    # plt.savefig('outputs/v4/edges/f1/Motif_' + m + '.png')
    # plt.savefig('outputs/v4/edges/f1/Motif_' + m + '.pdf')
    plt.show()
    plt.close()

if __name__ == "__main__":
    (graphConductance_experts, graphConductance_topUsers, titlesList) = \
        pickle.load(open('../data/DW_data/09_15/train/features/conductance_top0.2+experts.pickle', 'rb'))

    xtitles_list = []
    start_year = '2016'
    start_week = 14
    for idx in range(len(graphConductance_experts)):
        week = '(' + start_year + ', ' + str(start_week) + ')'
        xtitles_list.append(week)
        start_week += 1

    print(xtitles_list)
    plot_line(range(len(graphConductance_experts)), graphConductance_experts, graphConductance_topUsers, 'Experts', 'Top Users',xtitles_list )

    # print(len(gr))