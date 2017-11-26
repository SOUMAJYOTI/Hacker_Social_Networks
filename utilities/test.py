import pandas as pd
from pylab import *
import os
import math
import statistics as st
import statistics
from math import  *
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

def plot_bars(data, xTicks, xLabels='', yLabels=''):
    hfont = {'fontname': 'Arial'}
    ind = np.arange(len(data))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    width=0.35
    rects1 = ax.bar(ind, data, width,
                    color='black')  # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(87, 95)
    ax.set_ylabel(yLabels, size=30, **hfont)
    ax.set_xlabel(xLabels, size=30, **hfont)
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTicks, **hfont)
    plt.setp(xtickNames, rotation=45, fontsize=5)
    plt.grid(True)
    plt.xticks(size=20)
    plt.yticks(size=20)
    # plt.subplots_adjust(left=0.13, bottom=0.30, top=0.9)
    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()
    plt.close()

# plot_bars([0.13, 0.32, 0.36], ['Random', 'Without Exposure', 'Full Model'], yLabels='Accuracy')

def plot_line(x, y1, y2, y3, l1, l2, l3, ):
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111)  # row x col x position (here 1 x 1 x 1)

    # ax.imshow(subim, aspect='auto', extent=(5.8, 8.3, 0.8, 1), zorder=-1)
    # ax.add_patch(
    #     patches.Rectangle(
    #         (5.2, 0.77),
    #         2.5,
    #         0.28,
    #         fill=False,  # remove background
    #         linewidth=3
    #     )
    # )
    plt.xticks(x, size=30)  # rotate x-axis labels to 75 degree
    plt.yticks(size=30)
    ax.plot(x, y1,  marker='o', linestyle='-', label=l1, linewidth=3)
    ax.plot(x, y2, marker='o', linestyle='-', label=l2, linewidth=3)
    ax.plot(x, y3, marker='o', linestyle='-', label=l3, linewidth=3)
    # ax.plot(x, y4, marker='o', linestyle='-', label=l4, linewidth=3)
    # ax.plot(x, y5, marker='o', linestyle='-', label=l5, linewidth=3)

    # plt.xlim(0, len(var) + 1)
    plt.tight_layout()  # showing xticks (as xticks have long names)
    ax.grid()

    # plt.title(title, color='#000000', weight="bold", size=30)
    plt.ylabel('Accuracy', size=30)
    plt.xlabel('Number of unobserved nodes', size=30)

    ax.legend(loc='upper right', fancybox=True, shadow=True, fontsize=25)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.90)

    plt.show()
    # plt.ylim([0.05, 1.05])
    # plt.xlim([1.5, 10.5])
    # plt.savefig('v1/min_4/f1_plots_RF/Motif_' + m + '.png')
    # plt.savefig('v1/min_4/f1_plots_RF/Motif_' + m + '.pdf')
    plt.close()

x = [100, 300, 500, 700, 1000]
y1 = [0.13, 0.16, 0.12, 0.11, 0.12]
y2 = [0.29, 0.32, 0.24, 0.13, 0.09]
y3 = [0.26, 0.36, 0.28, 0.13, 0.1]

plot_line(x, y1, y2, y3, 'Random', 'Without Exposure', 'Full Model', )

