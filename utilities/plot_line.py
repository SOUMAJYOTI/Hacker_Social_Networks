import numpy as np
import matplotlib.pyplot as plt
from pylab import *


def plot_bars(data, xTicks, xLabels='', yLabels=''):
    hfont = {'fontname': 'Arial'}
    ind = np.arange(len(data))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    width=0.35
    rects1 = ax.bar(ind, data, width,
                    color='#0000ff')  # axes and labels
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


def plot_line(x, y, l, title, ):
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111)  # row x col x position (here 1 x 1 x 1)

    plt.xticks(size=30)  # rotate x-axis labels to 75 degree
    plt.yticks(size=30)

    x = np.array(range(len(x))) + 1
    # print(x.shape, len(y1))
    ax.plot(x, y,  marker='o', markersize=3, linestyle='-', linewidth=3, color='black')

    # plt.xlim(0, len(var) + 1)
    plt.tight_layout()  # showing xticks (as xticks have long names)
    ax.grid()

    # plt.title(title, color='#000000', weight="bold", size=30)
    plt.ylabel('Correlation', size=40)
    plt.xlabel('Lag order', size=40)

    # ax.legend(loc='lower right', fancybox=True, shadow=True, fontsize=25)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.90)
    plt.title(title, size=30)
    # plt.ylim([0, 1])
    # plt.xlim([0, 6])
    # plt.close()
    plt.show()


if __name__ == "__main__":
    x = range(8)
    x =  [i + 1 for i in x]
    y = [149313, 146388,	133789,	137738,	148130,	149335,	146625,	138573]
    title = 'Number of participating users'
    l = ''
    plot_line(x, y, l, title)
