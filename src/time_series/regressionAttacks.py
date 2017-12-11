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
from sklearn import linear_model, ensemble
from sklearn.naive_bayes import GaussianNB
from random import shuffle

from sklearn.svm import SVC
from sklearn import tree


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


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


def prepareData(inputDf, outputDf):
    y_actual = outputDf['attackFlag']

    # print(outputDf)
    train_start_date = outputDf.iloc[0, 0]
    train_end_date = outputDf.iloc[-1, 0]

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    delta_prev_time = 7  # no of days to check before the week of current day

    currDate = train_start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_actual.shape[0], delta_prev_time*num_features))
    Y = np.zeros((y_actual.shape[0], 1))

    while (currDate <= train_end_date):
        ''' This loop checks values on either of delta days prior'''
        for idx in range(delta_prev_time):
            historical_day = pd.to_datetime(currDate - datetime.timedelta(days=(14 - idx)))
            try:
                X[countDayIndx, idx*num_features:(idx+1)*num_features] = (inputDf[inputDf['date'] == historical_day.date()].ix[:,1:]).values[0]
            except:
                continue

        # print(outputDf[outputDf['date'] == currDate.date()])
        Y[countDayIndx] = [outputDf[outputDf['date'] == currDate.date()].ix[:, 1]]

        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    for col in range(X.shape[1]):
        X[np.where(X[:, col] == 0)[0], col] = np.median(X[:, col])
    # print(Y)
    # print(Y.shape)
    return X, Y


def anomalyVec(res_vec, ):
    mean_rvec = np.mean(res_vec)

    anomaly_vec = np.zeros(mean_rvec.shape)
    for t in range(res_vec.shape[0]):
        if res_vec[t] > mean_rvec:
            anomaly_vec[t] = 1.

    return anomaly_vec


def main():
    forums = [35, 38, 60, 62, 71, 84, 88, 105, 133, 135, 146, 147, 150, 161, 173, 179, 197, ]
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-03-01', '%Y-%m-%d')

    ''' Concatenat the features into a singale dataframe'''
    fileName_prefix = ['shortestPaths', 'conductance', 'commThreads']
    feat_df = pd.DataFrame()
    for fp in fileName_prefix:
        if feat_df.empty:
            feat_df = pd.read_pickle('../../data/DW_data/features/' + str(fp) + '_DeltaT_4_Sept16-Apr17_TP10.pickle')
        else:
            curr_df = pd.read_pickle('../../data/DW_data/features/' + str(fp) + '_DeltaT_4_Sept16-Apr17_TP10.pickle')
            feat_df = pd.merge(feat_df, curr_df, on=['date', 'forum'])

    feat_df = feat_df[feat_df['forum'].isin(forums)]

    subspace_df = pickle.load(open('../../data/DW_data/features/subspace_df_v12_10.pickle', 'rb'))
    instance_TrainStartDate = trainStart_date - relativedelta(months=1)
    trainDf = subspace_df[subspace_df['date'] >= instance_TrainStartDate]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]

    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
    y_actual_train = trainOutput['attackFlag']

    testStart_date = datetime.datetime.strptime('2017-03-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')

    instance_TestStartDate = testStart_date - relativedelta(months=1)
    testDf = subspace_df[subspace_df['date'] >= instance_TestStartDate]
    testDf = testDf[testDf['date'] < testEnd_date]

    '''   THIS TIMEFRAME IS IMPORTANT  !!!! '''
    testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)

    # y_random = np.array([random.choice([0, 1]) for _ in range(len(testOutput))])
    # print(testOutput)
    y_actual_test = list(testOutput['attackFlag'])
    y_random = y_actual_test.copy()

    shuffle(y_random)
    random_prec = sklearn.metrics.precision_score(y_actual_test, y_random)
    random_recall = sklearn.metrics.recall_score(y_actual_test, y_random)
    random_f1 = sklearn.metrics.f1_score(y_actual_test, y_random)
    print('Random: ', random_prec, random_recall, random_f1)

    ''' Plot the features forum wise '''
    features = ['shortestPaths', 'CondExperts', 'expertsThreads']
    X_all_train = []
    X_all_test = []

    # for feat in features:
    X_train, Y_train = prepareData(trainDf, trainOutput)
    X_test, Y_test = prepareData(testDf, testOutput)

    # df = pd.DataFrame(X)
    #
    # ## save to xlsx file
    #
    # filepath = '../../data/DW_data/features/feat_values_numpy/df_sp_ce_et_12_10_v1.csv'
    #
    # df.to_csv(filepath, index=False)
    # exit()
        # if X_all_train == []:
        #     X_all_train = X
        # else:
        #     X_all_train = np.concatenate((X_all_train, X), axis=1)
        #
        # logreg = linear_model.LogisticRegression(penalty='l2')
        # logreg = ensemble.RandomForestClassifier()
        # logreg.fit(X, y_actual_train)
        #
        # X_test = prepareData(testDf, testOutput, feat)
        # if X_all_test == []:
        #     X_all_test = X_test
        # else:
        #     X_all_test = np.concatenate((X_all_test, X_test), axis=1)
        # #
        # y_pred = logreg.predict(X_test)
        #
        # # # Attack prediction evaluation
        # prec, rec, f1_score = sklearn.metrics.precision_score(y_actual_test, y_pred),\
        #                       sklearn.metrics.recall_score(y_actual_test, y_pred), \
        #                       sklearn.metrics.f1_score(y_actual_test, y_pred),
        #
        # print(feat, prec, rec, f1_score)

    # clf = linear_model.LogisticRegression(penalty='l1')
    # clf = tree.DecisionTreeClassifier()
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, Y_train)

    # print(Y_train)
    # y_pred = logreg.predict(X_train)

    # clf = SVC(kernel='rbf')
    # clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)

    # print(y_pred)
    # Attack prediction evaluation
    prec, rec, f1_score = sklearn.metrics.precision_score(Y_test, y_pred),\
                          sklearn.metrics.recall_score(Y_test, y_pred), \
                          sklearn.metrics.f1_score(Y_test, y_pred),

    print(prec, rec, f1_score)

if __name__ == "__main__":
    main()

