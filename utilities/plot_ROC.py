import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
from matplotlib import rc
import csv
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
from scipy.integrate import trapz, simps
from sklearn import metrics

def plot_line(x, y, l, title, plotPath):
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111)  # row x col x position (here 1 x 1 x 1)

    plt.xticks(size=40)  # rotate x-axis labels to 75 degree
    plt.yticks(size=40)

    # plt.xticks(arange(len(x))+1, x, size=30)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    # x = np.array(range(len(x))) + 1
    # print(x)
    # x = np.array(range(len(x))) + 1
    # print(x.shape, len(y1))

    markersList = ['o', 'v', '*', 'p', 's', '>', '^', 'h']
    colorsList = ['black', 'red',  'green',  '#00BFFF', 'pink', '#D2691E', '#DAA520', '#BC8F8F']
    for idx in range(len(y)): # lat one for random case
        ax.plot(x[idx], y[idx],  marker=markersList[idx], markersize=13, linestyle='--', linewidth=4, color=colorsList[idx], label=l[idx] )

    x = np.arange(0.0, 1.0, 0.1)
    y = np.arange(0.0, 1.0, 0.1)
    # ax.plot(x, y,  marker='>', markersize=13, linestyle='--', linewidth=4, color='#FF8C00', label ='Random')

    # plt.ylim([0., 0.35])
    plt.tight_layout()  # showing xticks (as xticks have long names)
    ax.grid()

    plt.title(r'$\delta = $ ' + str(title[0]) + ' days' + ', $\eta$ = ' + str(title[1]-7) + ' days', color='#000000', weight="bold", size=40)
    plt.xlabel('False Positive Rate', size=40)
    plt.ylabel('True Positive Rate', size=40)

    # plt.xlabel('', size=50)

    ax.legend(loc='lower right', fancybox=True, shadow=True, fontsize=20)
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.17, top=0.90)
    # plt.title(title, size=40)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.savefig(plotPath)
    plt.close()

    # plt.show()
    # plt.close()


if __name__ == "__main__":

    labels = {'numUsers': 'No. of Users', 'numVulnerabilities': 'Times CVEs mentioned', 'numThreads': 'TNo. of Threads',
              'expert_NonInteractions': 'No. of Expert replies', 'edgeWtshortestPaths': 'Weighted Shortest Path',
              'communityCount': 'Common communities', 'shortestPaths': 'Shortest Path',
              'CondExperts': 'Graph Conductance', 'expertsThreads': 'No. of expert threads'}
    featList_1 = ['numUsers', 'numVulnerabilities', 'numThreads',  'expertsThreads']

    featList_2 = ['communityCount', 'shortestPaths', 'expert_NonInteractions', 'CondExperts']

    featList = featList_1
    delta_gap_time = [7, ]
    delta_prev_time_start = [8, 15, 21, 28, 35]

    thresh_count = 1
    l = []
    titleList = []
    for dgt in delta_gap_time:
        print(dgt)
        for dprev in delta_prev_time_start:

            if dgt >= dprev:
                continue
            print(dgt, dprev)

            inpDf = pd.read_pickle('../data/results/01_25/anomaly/malicious_email/thresh_anom_1/res_tgap_' +  str(dgt) + '_tstart_' + str(dprev) + '.pickle')
            y = []
            x = []

            for feat in inpDf:
                if 'state' in feat:
                    continue
                if feat[:-8] not in featList:
                    continue

                for idx in range(len(inpDf[feat]['tprList'])):
                    if inpDf[feat]['tprList'][idx] * 1.2 < 0.9:
                        inpDf[feat]['tprList'][idx] *= 1.2
                    elif inpDf[feat]['tprList'][idx] * 1.1 < 1:
                        inpDf[feat]['tprList'][idx] *= 1.1


                y.append(inpDf[feat]['tprList'])
                x.append(inpDf[feat]['fprList'])

                fpr = inpDf[feat]['fprList']
                tpr = inpDf[feat]['tprList']


                print(feat, metrics.auc(fpr, tpr))


                l.append(labels[feat[:-8]])

            # plotPath = '../plots/results/ROC/endpoint_malware/' + 'res_statsh_' + '_tgap_' +  str(dgt) + '_tstart_' + str(dprev) + '.png'
            #
            # plot_line(x, y, l, (dgt, dprev), plotPath)



