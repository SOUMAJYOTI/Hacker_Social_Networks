import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as o

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


def barplot(ax, dpoints, categories):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.

    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''

    # Aggregate the conditions and the categories according to their
    # mean values

    colors = ['#F5F5F5', '#D3D3D3', '#808080', '#000000']
    hatch_pat=['-', '+', 'x', '\\']
    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    # conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    # categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]

    conditions = ['precision', 'recall', 'F1', 'Random - F1']
    # categories = ['numUsers', 'numVulnerabilities', 'numThreads', 'expertsThreads']

    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    # Create a set of bars at each position
    for i, cond in enumerate(conditions):
        indeces = range(1, len(categories) + 1)
        vals = dpoints[dpoints[:, 0] == cond][:, 2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond,
               color=colors[i],
               edgecolor='black',
               # fill=False,
               )#cm.Accent(float(i) / n))

    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces,)
    # ax.set_xticklabels(categories)
    # plt.setp(plt.xticks()[1], rotation=60)

    # Add the axis labels
    ax.set_ylabel("", size=40, **hfont)
    # ax.set_xlabel("Structure")

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=22)

    textstr = '1- ' + categories[0] + ', 2 - ' + categories[1] + ', 3 - ' + categories[2] + ', 4 - ' + categories[3]

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    font0 = FontProperties()
    plt.subplots_adjust(left=0.18, bottom=0.17, top=0.85, right=0.9)
    # font0.set_size('large')
    font0.set_weight('bold')

    # place a text box in upper left in axes coords
    ax.text(0., 1.1, textstr, transform=ax.transAxes, fontsize=14, fontproperties=font0, verticalalignment='top', bbox=props)


if __name__ =="__main__":


    labels = {'numUsers': 'No. of Users', 'numVulnerabilities': 'Times CVEs mentioned', 'numThreads': 'No. of Threads',
              'expert_NonInteractions': 'No. of Expert replies', 'edgeWtshortestPaths': 'Weighted Shortest Path',
              'communityCount': 'Common communities', 'shortestPaths': 'Graph Conductance',
              'CondExperts': 'Shortest Path', 'expertsThreads': 'No. of expert threads'}
    # featList = ['numUsers', 'numVulnerabilities', 'numThreads', 'expert_NonInteractions', 'edgeWtshortestPaths',
    #             'communityCount', 'shortestPaths', 'CondExperts', 'expertsThreads']

    featList = ['numUsers', 'numVulnerabilities', 'numThreads', 'expertsThreads']
    categories = []
    delta_gap_time = [7,  ]
    delta_prev_time_start = [ '8', ]

    # y = []
    l = ['precision', 'recall', 'F1']
    categories = []
    for dgt in delta_gap_time:
        print(dgt)
        inpDf = pd.read_pickle('../data/results/01_25/regular/LR_L2/tgap_' + str(dgt) + '.pickle')

        row_array = []

        for feat in inpDf:
            if feat not in featList:
                continue
            print(feat)
            categories.append(labels[feat])
            for idx in range(len(delta_prev_time_start)):
                row_array.append(['precision', labels[feat], inpDf[feat]['precision'][idx]] )

            for idx in range(len(delta_prev_time_start)):
                row_array.append(['recall', labels[feat], inpDf[feat]['recall'][idx]] )

            for idx in range(len(delta_prev_time_start)):
                row_array.append(['F1', labels[feat], inpDf[feat]['f1'][idx]] )

            for idx in range(len(delta_prev_time_start)):
                row_array.append(['Random - F1', labels[feat], 0.37] )

            plotPath = '../../plots/results/classification/' + feat + '_tgap_' + str(dgt) + '.png'

        # print(row_array)
        # print(np.array(row_array).shape, dpoints_inhib.shape)
        hfont = {'fontname': 'Arial'}
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        plt.tick_params('x', labelsize=30)
        plt.tick_params('y', labelsize=30)
        plt.grid(True)
        plt.rc('legend', **{'fontsize': 8})
        plt.subplots_adjust(left=0.13, bottom=0.15, top=0.9)
        plt.ylim([0, 0.9])
        # plt.xlabel(r'Time Lag $\delta $ (in days)', size=40)
        # plt.title(labels[feat], size=40)

        barplot(ax, np.array(row_array), categories)

        plt.show()
        # plt.savefig(plotPath)
        plt.close()

        # savefig('barchart_3.png')
        # plt.show()