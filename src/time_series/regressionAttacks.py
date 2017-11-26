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


def prepareData(inputDf, outputDf, feat):
    y_actual = outputDf['attackFlag']
    train_start_date = outputDf.iloc[0, 0]
    train_end_date = outputDf.iloc[-1, 0]

    delta_prev_time = 1  # no of days to check before the week of current day

    currDate = train_start_date
    countDayIndx = 0

    X = np.zeros((y_actual.shape[0], delta_prev_time))

    while (currDate <= train_end_date):
        ''' This loop checks values on either of delta days prior'''
        for idx in range(delta_prev_time):
            historical_day = currDate - datetime.timedelta(days=(7 - idx))
            try:
                X[countDayIndx, idx] = inputDf[inputDf['date'] == historical_day][feat]
            except:
                continue
        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    return X


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

    trainStart_date = datetime.datetime.strptime('2016-9-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-03-01', '%Y-%m-%d')

    feat_df = pickle.load(open('../../data/DW_data/feature_df_combined_Sept16-Apr17.pickle', 'rb'))

    trainDf = feat_df[feat_df['date'] >=trainStart_date.date()]
    trainDf = trainDf[trainDf['date'] < trainEnd_date.date()]

    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
    y_actual_train = trainOutput['attackFlag']

    testStart_date = datetime.datetime.strptime('2017-02-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')

    testDf = feat_df[feat_df['date'] >= testStart_date.date()]
    testDf = testDf[testDf['date'] < testEnd_date.date()]

    '''   THIS TIMEFRAME IS IMPORTANT  !!!! '''
    testOutput = prepareOutput(amEvents_malware, testStart_date + relativedelta(months=1),
                               testEnd_date)

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
    features = ['conductance', 'conductanceExperts', 'pagerank', 'degree']
    X_all_train = []
    X_all_test = []

    for feat in features:
        #     dir_save = '../../plots/dw_stats/feat_plot/' + str(feat) + '/'
        #     if not os.path.exists(dir_save):
        #         os.makedirs(dir_save)
        #     for f in forums:
        #         plotDf = trainDf[trainDf['forum'] == f]
        #         plotDf.plot(figsize=(12,8), x='date', y=feat, color='black', linewidth=2)
        #         plt.grid(True)
        #         plt.xticks(size=20)
        #         plt.yticks(size=20)
        #         plt.xlabel('date', size=20)
        #         plt.ylabel(feat, size=20)
        #         plt.title('Forum=' + str(f))
        #         plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
        #         # plt.show()
        #         file_save = dir_save + 'forum_' + str(f)
        #         plt.savefig(file_save)
        #         plt.close()

        X = prepareData(trainDf, trainOutput, feat)
        if X_all_train == []:
            X_all_train = X
        else:
            X_all_train = np.concatenate((X_all_train, X), axis=1)
        logreg = linear_model.LogisticRegression(penalty='l2')
        logreg = ensemble.RandomForestClassifier()
        logreg.fit(X, y_actual_train)

        X_test = prepareData(testDf, testOutput, feat)
        if X_all_test == []:
            X_all_test = X_test
        else:
            X_all_test = np.concatenate((X_all_test, X_test), axis=1)
        #
        y_pred = logreg.predict(X_test)

        # # Attack prediction evaluation
        prec, rec, f1_score = sklearn.metrics.precision_score(y_actual_test, y_pred),\
                              sklearn.metrics.recall_score(y_actual_test, y_pred), \
                              sklearn.metrics.f1_score(y_actual_test, y_pred),

        print(feat, prec, rec, f1_score)

    logreg = ensemble.RandomForestClassifier()
    logreg.fit(X_all_train, y_actual_train)

    y_pred = logreg.predict(X_all_test)

    # Attack prediction evaluation
    prec, rec, f1_score = sklearn.metrics.precision_score(y_actual_test, y_pred),\
                          sklearn.metrics.recall_score(y_actual_test, y_pred), \
                          sklearn.metrics.f1_score(y_actual_test, y_pred),

    print(prec, rec, f1_score)

if __name__ == "__main__":
    main()

