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
from sklearn import preprocessing
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
    train_start_date = outputDf.iloc[0, 0] # first row
    train_end_date = outputDf.iloc[-1, 0] # last row

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    y_true = outputDf['attackFlag']
    delta_prev_time = 4  # no of days to check before the week of current day

    currDate = train_start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_true.shape[0], delta_prev_time*num_features)) #input
    Y = np.zeros((y_true.shape[0], 1)) #output

    while (currDate <= train_end_date):
        ''' This loop checks values on either of delta days prior '''
        for idx in range(delta_prev_time):
            historical_day = pd.to_datetime(currDate - datetime.timedelta(days=(14 - idx))) # one week before
            try:
                X[countDayIndx, idx*num_features:(idx+1)*num_features] = (inputDf[inputDf['date'] == historical_day.date()].ix[:,1:]).values[0] # exclude date
            except:
                continue

        # print(outputDf[outputDf['date'] == currDate.date()])
        Y[countDayIndx] = [outputDf[outputDf['date'] == currDate.date()].ix[:, 1]]

        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    # Fill the zero cells with median of column
    for col in range(X.shape[1]):
        X[np.where(X[:, col] == 0)[0], col] = np.median(X[~np.where(X[:, col] == 0)[0], col])

    return X, Y


def plot_feat_stats(df):
    df.hist(color='black')
    plt.show()
    plt.grid(True)
    plt.xticks(size=10)
    plt.yticks(size=10)
    # plt.title(size=10)
    # plt.ylabel('Attack/Not attack', size=20)
    # plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    # # file_save = dir_save + 'CPE_R' + str(idx_cpe+1)
    # # plt.savefig(file_save)
    # plt.close()


def main():
    forums = [35, 38, 60, 62, 71, 84, 88, 105, 133, 135, 146, 147, 150, 161, 173, 179, 197, ]
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-04-01', '%Y-%m-%d')

    subspace_df = pickle.load(open('../../data/DW_data/features/feat_combine/user_interStats_DeltaT_4_Sept16-Apr17_TP10.pickle', 'rb'))

    # subspace_df = subspace_df.ix[:, :6]
    instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features
    trainDf = subspace_df[subspace_df['date'] >= instance_TrainStartDate]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]

    # plot_feat_stats(trainDf)

    # print(trainDf.shape)
    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
    # y_actual_train = trainOutput['attackFlag']

    testStart_date = datetime.datetime.strptime('2017-04-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')

    instance_TestStartDate = testStart_date - relativedelta(months=1)
    testDf = subspace_df[subspace_df['date'] >= instance_TestStartDate]
    testDf = testDf[testDf['date'] < testEnd_date]

    testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)

    # y_random = np.array([random.choice([0, 1]) for _ in range(len(testOutput))])
    # print(testOutput)
    y_actual_test = list(testOutput['attackFlag'])
    # y_random = y_actual_test.copy()

    # shuffle(y_random)

    y_random = np.random.randint(2, size=len(y_actual_test))
    random_prec = sklearn.metrics.precision_score(y_actual_test, y_random)
    random_recall = sklearn.metrics.recall_score(y_actual_test, y_random)
    random_f1 = sklearn.metrics.f1_score(y_actual_test, y_random)
    print('Random: ', random_prec, random_recall, random_f1)

    X_train, Y_train = prepareData(trainDf, trainOutput)
    X_test, Y_test = prepareData(testDf, testOutput)

    # X_train = preprocessing.normalize(X_train, norm='l2')
    # X_test = preprocessing.normalize(X_test, norm='l2')

    # print(X_train)
    clf = linear_model.LogisticRegression(penalty='l1', class_weight='balanced')
    # clf = tree.DecisionTreeClassifier()
    # clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, Y_train)
    # print(clf.coef_.shape)
    # for coef_idx in range(clf.coef_.shape[1]):
    #     if clf.coef_[0][coef_idx] < 0.1 and clf.coef_[0][coef_idx] >= 0:
    #         clf.coef_[0][coef_idx] = 0.
    # print(clf.coef_)

    # print(Y_train)
    # y_pred = logreg.predict(X_train)

    # clf = SVC(kernel='rbf', C=1000.1)
    # clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)

    y_test_pred = []
    prob = clf.predict_proba(X_test)
    for r in range(prob.shape[0]):
        # print(X_test[r][1])
        if prob[r][1] > 0.5:
            y_test_pred.append(1)
        else:
            y_test_pred.append(0)
    # print(np.count_nonzero(Y_train) / Y_train.shape[0])
    # print(np.count_nonzero(Y_test) / Y_test.shape[0])
    print(Y_test.flatten())
    print(y_random)
    print(y_test_pred)

    # testInpOut_df = pd.DataFrame()
    # testInpOut_df['date'] =np.arange('2017-03', '2017-05', dtype='datetime64[D]')
    #     # np.arange('2017-03-01',
    #     #                              '2017-05-01', dtype = 'datetime')
    # testInpOut_df['actual_attack'] = Y_test
    # testInpOut_df['predicted_attack'] = y_pred

    # print(y_pred)
    # Attack prediction evaluation
    prec, rec, f1_score = sklearn.metrics.precision_score(Y_test, y_test_pred),\
                          sklearn.metrics.recall_score(Y_test, y_test_pred), \
                          sklearn.metrics.f1_score(Y_test, y_test_pred),

    print(prec, rec, f1_score)

if __name__ == "__main__":
    main()

