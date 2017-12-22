import pandas as pd
import datetime as dt
import operator
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import os
from dateutil.relativedelta import relativedelta
import sklearn.metrics
import random
from random import shuffle
from sklearn import linear_model, ensemble
from sklearn.naive_bayes import GaussianNB
pd.set_option("display.precision", 2)
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

forumsList = [35, 38, 133, 135, 146,  150, 161, 197, ]

class ArgsStruct:
    name = ''
    feat_concat = False
    forumsSplit = False
    plot_data = False
    plot_time_series = False
    plot_subspace = False


def prepareOutput(eventsDf, start_date, end_date):
    eventsDf['date'] = pd.to_datetime(eventsDf['date'])

    # For now, just consider the boolean case of attacks, later can extend to count
    currDay = start_date
    datesList = []
    attackFlag = []
    while(currDay < end_date):
        try:
            events = eventsDf[eventsDf['date'] == currDay]
            total_count = pd.DataFrame(events.groupby(['date']).sum())
            count_attacks = (total_count['count'].values)[0]

            if count_attacks == 0:
                attackFlag.append(0)
            else:
                attackFlag.append(1)
        except:
            attackFlag.append(0)

        datesList.append(currDay)
        currDay = currDay + datetime.timedelta(days=1)

    outputDf = pd.DataFrame()
    outputDf['date'] = datesList
    outputDf['attackFlag'] = attackFlag

    return outputDf


def featureAnalysis(feat_df, output_df, trainStart_date, delta_time_prev):

    ''' For the attacks history'''
    attacks_df = output_df[output_df['attackFlag'] == 1]

    feat_hist_attacks_array = {}

    # print(feat_df[:40])
    for idx, row in attacks_df.iterrows():
        attack_date = row['date']
        time_hist = attack_date

        for idx_time in range(1, delta_time_prev+1):
            time_hist = pd.to_datetime(time_hist - datetime.timedelta(days=1))

            feat_vals = feat_df[feat_df['date'] == time_hist]
            while feat_vals.empty == True:
                # print(time_hist)
                time_hist = time_hist - datetime.timedelta(days=1)
                feat_vals = feat_df[feat_df['date'] == time_hist]
                # print(feat_vals)

            features = list(feat_vals.columns.values)

            for idx_f in range(1, len(features)): # exclude date
                if features[idx_f] not in feat_hist_attacks_array:
                    feat_hist_attacks_array[features[idx_f]] = [[] for _ in range(delta_time_prev)]

                # print(feat_vals)
                feat_hist_attacks_array[features[idx_f]][idx_time-1].append(feat_vals.iloc[0, idx_f])

    ''' For the non attacks history '''
    non_attacks_df = output_df[output_df['attackFlag'] == 0]
    non_attacks_df = non_attacks_df[non_attacks_df['date'] >= trainStart_date]

    feat_hist_non_attacks_array = {}

    # print(feat_df[:40])
    for idx, row in non_attacks_df.iterrows():
        non_attack_date = row['date']
        time_hist = non_attack_date

        # print(non_attack_date)
        for idx_time in range(1, delta_time_prev + 1):
            time_hist = pd.to_datetime(time_hist - datetime.timedelta(days=1))
            # print(time_hist)

            feat_vals = feat_df[feat_df['date'] == time_hist]
            while feat_vals.empty == True:
                # print(time_hist)
                time_hist = time_hist - datetime.timedelta(days=1)
                feat_vals = feat_df[feat_df['date'] == time_hist]
                # print(feat_vals)

            features = list(feat_vals.columns.values)

            for idx_f in range(1, len(features)):  # exclude date
                if features[idx_f] not in feat_hist_non_attacks_array:
                    feat_hist_non_attacks_array[features[idx_f]] = [[] for _ in range(delta_time_prev)]

                # print(feat_vals)
                feat_hist_non_attacks_array[features[idx_f]][idx_time - 1].append(feat_vals.iloc[0, idx_f])

    return feat_hist_attacks_array, feat_hist_non_attacks_array


def featureAnalysis_forums(feat_df, output_df, trainStart_date, forum, delta_time_prev):

    ''' For the attacks history'''
    attacks_df = output_df[output_df['attackFlag'] == 1]
    non_attacks_df = output_df[output_df['attackFlag'] == 0]
    non_attacks_df = non_attacks_df[non_attacks_df['date'] >= trainStart_date]


    feat_hist_attacks_array = {}
    feat_hist_non_attacks_array = {}

    feat_forum = feat_df[feat_df['forum'] == forum]

    for idx, row in attacks_df.iterrows():
        attack_date = row['date']
        time_hist = attack_date

        for idx_time in range(1, delta_time_prev+1):
            time_hist = pd.to_datetime(time_hist - datetime.timedelta(days=1))

            feat_vals = feat_forum[feat_forum['date'] == time_hist]
            while feat_vals.empty == True:
                # print(time_hist)
                time_hist = time_hist - datetime.timedelta(days=1)
                feat_vals = feat_forum[feat_forum['date'] == time_hist]
                # print(feat_vals)

            features = list(feat_vals.columns.values)

            for idx_f in range(2, len(features)): # exclude date and forum
                if features[idx_f] not in feat_hist_attacks_array:
                    feat_hist_attacks_array[features[idx_f]] = [[] for _ in range(delta_time_prev)]

                # print(feat_vals)
                feat_hist_attacks_array[features[idx_f]][idx_time-1].append(feat_vals.iloc[0, idx_f])

    ''' For the non attacks history '''
    for idx, row in non_attacks_df.iterrows():
        non_attack_date = row['date']
        time_hist = non_attack_date

        # print(non_attack_date)
        for idx_time in range(1, delta_time_prev + 1):
            time_hist = pd.to_datetime(time_hist - datetime.timedelta(days=1))
            # print(time_hist)

            feat_vals = feat_forum[feat_forum['date'] == time_hist]
            while feat_vals.empty == True:
                # print(time_hist)
                time_hist = time_hist - datetime.timedelta(days=1)
                feat_vals = feat_forum[feat_forum['date'] == time_hist]
                # print(feat_vals)

            features = list(feat_vals.columns.values)

            for idx_f in range(2, len(features)):  # exclude date anf forum
                if features[idx_f] not in feat_hist_non_attacks_array:
                    feat_hist_non_attacks_array[features[idx_f]] = [[] for _ in range(delta_time_prev)]

                # print(feat_vals)
                feat_hist_non_attacks_array[features[idx_f]][idx_time - 1].append(feat_vals.iloc[0, idx_f])

    return feat_hist_attacks_array, feat_hist_non_attacks_array


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


def concatenateDf(featList):
    '''
    Concatenate the features into a single dataframe

    :param df_list:
    :param featList:
    :return:
    '''

    feat_df = pd.DataFrame()
    for fp in featList:
        if feat_df.empty:
            feat_df = pd.read_pickle('../../data/DW_data/features/feat_forums/'
                                     + str(fp) + '_DeltaT_2_Sept16-Apr17.pickle')
        else:
            curr_df = pd.read_pickle('../../data/DW_data/features/feat_forums/'
                                     + str(fp) + '_DeltaT_2_Sept16-Apr17.pickle')
            feat_df = pd.merge(feat_df, curr_df, on=['date'])

    return feat_df


# def time_to_event():



def plot_ts(df, plot_dir, title):
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
        file_save = plot_dir + feat + title + '.png'
        plt.savefig(file_save)
        plt.close()


def main():
    ''' Set the arguments for the data preprocessing here '''
    args = ArgsStruct()
    args.feat_concat = False
    args.forumsSplit = True
    args.plot_data = False
    args.plot_time_series = False
    args.plot_subspace = True

    trainStart_date = datetime.datetime.strptime('2016-9-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')

    trainInst_start = trainStart_date + relativedelta(months=1)


    ''' Load the organization attack data '''
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']
    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)

    if args.feat_concat == False:
        feat_df_graph = pd.read_pickle(
            '../../data/DW_data/features/feat_forums/graph_stats_Forums_Delta_T0_Sept16-Apr17.pickle')
        feat_df_users = pd.read_pickle(
            '../../data/DW_data/features/feat_forums/user_interStats_Forums_Delta_T0_Sept16-Apr17.pickle')

        feat_df_users = feat_df_users.filter(items=['date', 'forum', 'expert_NonInteractions', 'expertsInteractions', 'numUsers'])
        feat_df = pd.merge(feat_df_users, feat_df_graph, on=['date', 'forum'])
        pickle.dump(feat_df, open('../../data/DW_data/features/feat_forums/user_graph_Delta_T0_Sept16-Apr17.pickle', 'wb'))
        # print(feat_df)
    else:
        features = ['shortestPaths', 'conductance', 'commThreads']
        feat_df = concatenateDf(features)

    train_featDf = feat_df[feat_df['date'] >= trainStart_date]
    train_featDf = train_featDf[train_featDf['date'] < trainEnd_date]

    ''' Some plot functionalities '''
    delta_time_prev = 14
    if args.plot_data == True:
        if args.forumsSplit == False:
            attack_hist, non_attack_hist = featureAnalysis(train_featDf, trainOutput, trainInst_start, delta_time_prev)
            title = ''
            plot_dir = '../../plots/dw_stats/feat_plot/feat_combine/feature_distribution/graph_stats/'
            plot_box_dist(attack_hist, non_attack_hist, title, plot_dir, delta_time_prev)
        else:
            for f in forumsList:
                attack_hist, non_attack_hist = featureAnalysis_forums(train_featDf, trainOutput,  trainInst_start, f, delta_time_prev)
                title = 'Forum_' + str(f)
                plot_dir = '../../plots/dw_stats/feat_plot/feat_forums/feature_distribution/user_stats/'
                plot_box_dist(attack_hist, non_attack_hist, title, plot_dir, delta_time_prev)


    if args.plot_time_series == True:
        if args.forumsSplit == False:
            plot_dir = '../../plots/dw_stats/feat_plot/feat_combine/time_series/graph_stats/'
            plot_ts(train_featDf, plot_dir)
        else:
            for f in forumsList:
                forum_feat_df = train_featDf[train_featDf['forum'] == f]
                plot_dir = '../../plots/dw_stats/feat_plot/feat_forums/time_series/graph_stats/'
                title = 'Forum_' + str(f)
                plot_ts(forum_feat_df, plot_dir, title)


    if args.plot_subspace == True:
        subspace_df = pd.read_pickle('../../data/DW_data/features/subspace_df_v12_22.pickle')
        title = ''
        plot_dir = '../../plots/dw_stats/feat_plot/feat_combine/time_series/subspace/'
        plot_ts(subspace_df, plot_dir, title)

if __name__ == "__main__":
    main()
