import pandas as pd
import datetime as dt
import operator
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import os
from dateutil.relativedelta import relativedelta
pd.set_option("display.precision", 2)
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes


class ArgsStruct:
    name = ''
    LOAD_DATA = True
    feat_concat = True
    forumsSplit = False
    plot_data = False
    plot_time_series = True
    plot_subspace = False
    TYPE_PLOT = 'LINE'  ''' BAR: bar plot, LINE: line plot, STEM: stem plot'''


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_box_dist(attack_hist, non_attack_hist, title, dir, delta_time_prev):
    x_labels = range(1, delta_time_prev+1)

    for feat in attack_hist:
        plot_dir = dir + feat + '/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        fig = plt.figure(1, figsize=(20, 16))

        # Create an axes instance
        ax = fig.add_subplot(111)
        attack_data = [[] for _ in range(delta_time_prev)]
        non_attack_data = [[] for _ in range(delta_time_prev)]

        attack_xpos = []
        non_attack_xpos = []
        pos_index = 1
        x_tickPos = []
        for idx_time in range(len(attack_hist[feat])):
            # print(len(attack_hist[feat][idx_time]), len(non_attack_hist[feat][idx_time]))
            attack_data[idx_time].append(attack_hist[feat][idx_time])
            non_attack_data[idx_time].append(non_attack_hist[feat][idx_time])
            attack_xpos.append(pos_index)
            non_attack_xpos.append(pos_index+1)
            x_tickPos.append((pos_index + pos_index + 1)/2)
            pos_index += 2

        bpl = plt.boxplot(attack_data, positions=attack_xpos, sym='', widths=0.6)
        bpr = plt.boxplot(non_attack_data, positions=non_attack_xpos, sym='', widths=0.6)
        set_box_color(bpl, 'red')  # colors are from http://colorbrewer2.org/
        set_box_color(bpr, 'blue')

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='red', label='Attacks')
        plt.plot([], c='blue', label='Non attacks')
        plt.legend(fontsize=30, loc='upper right')

        # set axes limits and labels
        # xlim(0, pos_index)
        # ylim(0, pos_index)

        ax.set_xticklabels(x_labels)
        ax.set_xticks(x_tickPos)

        plt.tick_params('y', labelsize=20)
        plt.tick_params('x', labelsize=20)
        # ax.set_yticks(fontsize=15)
        plt.title(title, size=25)

        plt.grid(True)
        plt.xlabel('Number of days prior to date of event', fontsize=30)
        plt.ylabel(feat, fontsize=30)
        # plt.show()
        plt.savefig(plot_dir + feat + '_' + title + '.png')
        plt.close()


def plot_ts(df, plot_dir, title):
    # print(df[:10])
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for feat in df.columns.values:
        if feat == 'date' or feat == 'forum':
            continue

        df.plot(figsize=(12,8), x='date', y=feat, color='black', linewidth=2)
        plt.grid(True)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title(title, size=20)
        plt.xlabel('date', size=20)
        plt.ylabel(feat, size=20)
        plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
        file_save = plot_dir + feat  + '.png'
        plt.savefig(file_save)
        plt.close()


def plot_ts_anomalies_bar(df, plot_dir, title):
    # print(df[:10])
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    labels = {'numUsers': '# Users', 'numVulnerabilities': '# CEVs mentioned', 'numThreads': '# Threads',
              'expert_NonInteractions': '# Expert replies', 'edgeWtshortestPaths': 'Weighted Shortest Path',
                'communityCount': '# common communities', 'shortestPaths': 'Shortest Path', 'CondExperts': 'Graph Conductance'}
    featList = ['numUsers', 'numVulnerabilities', 'numThreads', 'expert_NonInteractions', 'edgeWtshortestPaths',
                'communityCount', 'shortestPaths', 'CondExperts']

    for feat in df.columns.values:
        # if feat == 'start_dates' or feat == 'end_dates':
        #     continue

        # featName = feat
        if 'state' in feat:
            featName = feat[:-11]
        else:
            featName = feat[:-9]

        if featName not in featList:
            continue

        print(feat, featName)

        fig = plt.figure(1, figsize=(12, 8))
        ax = plt.subplot(111)  # row x col x position (here 1 x 1 x 1)

        y = np.array(df[feat])
        x = list(range(y.shape[0]))

        plt.bar(x, y, linestyle = '-',  lw=1, color='gray')

        plt.grid(True)
        dates = np.array(df['start_dates'])
        tick_pos = list(np.arange(min(x), max(x) + 1, 10.0).astype(int))
        date_pos = list(dates[tick_pos])
        # print(date_pos)

        date_pos = [str(d)[:7] for d in date_pos]
        plt.xticks(tick_pos, list(date_pos), rotation=90, )

        plt.xticks(size=30)
        plt.yticks(size=30)
        # plt.locator_params(axis='x', nticks=10)

        title = 'All Forums'
        # if 'state' in feat:
        #     title = 'State Vector'
        # else:
        #     title = 'Residual Vector'
        # plt.title(title, size=40)
        # plt.xlabel('date', size=40)
        plt.ylabel(labels[featName], size=40)
        plt.subplots_adjust(left=0.21, right=0.95,  bottom=0.25, top=0.85)
        # plt.show()
        file_save = plot_dir + feat + '.png'
        plt.savefig(file_save)
        plt.close()


def plot_ts_bar(df, plot_dir, title):
    # print(df[:10])
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for feat in df.columns.values:
        if feat == 'date' or feat == 'forum':
            continue

        fig, ax = plt.subplots()
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.date
        df.plot.bar(figsize=(12,8), ax=ax, x='date', y=feat, color='black', linewidth=2)
        plt.grid(True)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title(title, size=20)
        plt.xlabel('date', size=20)
        plt.ylabel(feat, size=20)
        plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
        file_save = plot_dir + feat  + '.png'
        plt.savefig(file_save)
        plt.close()


def main():
    ''' Set the arguments for the data preprocessing here '''
    args = ArgsStruct()
    args.plot_data = False
    args.plot_time_series = False
    args.plot_subspace = False
    args.TYPE_PLOT = 'LINE'

    trainStart_date = datetime.datetime.strptime('2016-03-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    anomaly_events = pd.read_pickle('../data/DW_data/features/feat_forums/anomaly_event_corr.pickle')
    # print(anomaly_events)

    plot_dir = '../plots/dw_stats/stats/anomaly/'
    plot_ts_anomalies_bar(anomaly_events, plot_dir, '')


if __name__ == "__main__":
    main()

