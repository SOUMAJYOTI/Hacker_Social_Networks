import sys
sys.path.append('../test')
sys.path.append('../lib')
from pyglmnet import GLM
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import sklearn.metrics
from sklearn import linear_model, ensemble
from sklearn.naive_bayes import GaussianNB
from random import shuffle
import csv

from sklearn.svm import SVC
from sklearn import tree
from sklearn import preprocessing
# import glmnet_python
# from glmnet import glmnet


class ArgsStruct:
    LOAD_DATA = True

'''

The purpose of the code is to compute the precision metrics for the supervised
and the unsupervised losses for the high attack weeks.

'''


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


def prepareData(inputDf, outputDf, delta_gap_time, delta_prev_time_start):
    start_date = pd.to_datetime(outputDf.iloc[0, 0]) # first row
    end_date = pd.to_datetime(outputDf.iloc[-1, 0]) # last row

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    y_true = outputDf['attackFlag']

    # This can be validated through
    # delta_gap_time = 7  # no of days to check
    # delta_prev_time_start = 28 # no of days to check before the current day

    currDate = start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_true.shape[0], delta_gap_time*num_features)) #input
    Y = -np.ones((y_true.shape[0], 1)) #output


    while (currDate <= end_date):
        ''' This loop checks values on either of delta days prior '''
        for idx in range(delta_gap_time):
            historical_day = pd.to_datetime(currDate - datetime.timedelta(days=delta_prev_time_start - idx)) # one week before
            try:
                X[countDayIndx, idx*num_features:(idx+1)*num_features] = (inputDf[inputDf['date'] == historical_day.date()].ix[:,1:]).values[0] # exclude date
            except:
                continue

        Y[countDayIndx] = (outputDf[outputDf['date'] == currDate.date()].ix[:, 1])

        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    # Fill the missing cells with median of column
    # for col in range(X.shape[1]):
    #     X[np.where(X[:, col] == 0)[0], col] = np.median(X[~np.where(X[:, col] == -1)[0], col])

    return X, Y,


def predictAttacks_onAnomaly(inputDf, outputDf, delta_gap_time, delta_prev_time_start, thresholds_anom
                             , feat, thresh_anom_count):
    '''

    :param inputDf:
    :param outputDf:
    :param delta_gap_time:
    :param delta_prev_time_start:
    :param thresholds_anom:
    :param feat: the feature in consideration
    :param thresh_anom_count:
    :return:
    '''

    ''' Metrics for ROC curve '''
    tprList = []
    fprList = []

    y_actual = outputDf['attackFlag']

    test_start_date = outputDf.iloc[0, 0]
    test_end_date = outputDf.iloc[-1, 0]

    week_predDf = pd.DataFrame()
    dateList = []
    ''' Iterate over all the thresholds for ROC curves '''
    for t in thresholds_anom:
        # print('Threshold: ', t)
        currDate = test_start_date
        y_estimate = np.zeros(y_actual.shape)
        countDayIndx = 0
        while(currDate <= test_end_date):
            '''This loop checks values on either of delta days prior'''
            count_anomalies = 0 # number of anomalies in the delta_prev time window
            for idx in range(delta_gap_time):
                historical_day = pd.to_datetime(currDate - datetime.timedelta(days=delta_prev_time_start - idx))

                try:
                    res_vec_value = inputDf[inputDf['date'] == historical_day][feat]
                except KeyError:
                    continue

                try:
                    if res_vec_value.iloc[0] > t:
                        count_anomalies += 1
                except:
                    # print(currDate)
                    continue

            ''' Choose threshold scaling more wisely !!!! '''
            ''' If the number of anomalies crosses a threshold over the time gap, predict as attack'''
            # print(count_anomalies)
            if count_anomalies > int((delta_gap_time/7 * thresh_anom_count)):
                y_estimate[countDayIndx] = 1

            countDayIndx += 1
            dateList.append(currDate)
            currDate += datetime.timedelta(days=1)



    return week_predDf


def main():
    args = ArgsStruct()
    args.LOAD_DATA = True
    args.forumsSplit = False
    args.FEAT_TYPE = "REGULAR" # REGULAR: graph features, ANOM: anomaly features
    args.TYPE_PLOT = 'LINE'

    ''' SET THE TRAINING AND TEST TIME PERIODS - THIS IS MANUAL '''
    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')

    testStart_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    if args.LOAD_DATA:
        amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
        amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

        # print(amEvents_malware)
        if args.FEAT_TYPE == "REGULAR":
            feat_df = pickle.load(open('../../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle', 'rb'))
        else:
            feat_df = pickle.load(
                open('../../data/DW_data/features/feat_combine/user_interStats_Delta_T0_Sep16-Aug17.pickle', 'rb'))

        # Prepare the dataframe for output
        trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)

        # the previous month is needed for features - since we measure lag correlation for the prediction
        instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features

        # Make the date common for all
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        trainDf = feat_df[feat_df['date'] >= instance_TrainStartDate]
        trainDf = trainDf[trainDf['date'] < trainEnd_date]

        # y_actual_train = list(trainOutput['attackFlag'])
        # y_actual_train = np.array(y_actual_train)
        # print(y_actual_train[y_actual_train == 1.].shape, y_actual_train.shape)

        instance_TestStartDate = testStart_date - relativedelta(months=1)
        testDf = feat_df[feat_df['date'] >= instance_TestStartDate]
        testDf = testDf[testDf['date'] < testEnd_date]
        testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)
        y_actual_test = list(testOutput['attackFlag'])


        ''' This is the random prediction without any priors'''
        # prec_rand = 0.
        # rec_rand = 0.
        # f1_rand = 0.
        # for idx_rand in range(5):
        #     y_random = np.random.randint(2, size=len(y_actual_test))
        #     # print(y_random)
        #     y_actual_test = np.array(y_actual_test)
        #     # print(y_actual_test[y_actual_test == 1.].shape, y_actual_test.shape)
        #     prec_rand += sklearn.metrics.precision_score(y_actual_test, y_random)
        #     rec_rand += sklearn.metrics.recall_score(y_actual_test, y_random)
        #     f1_rand += sklearn.metrics.f1_score(y_actual_test, y_random)
        #
        # prec_rand /= 5
        # rec_rand /= 5
        # f1_rand /= 5
        # print('Random: ', prec_rand, rec_rand, f1_rand)
        #

        ''' This is the prediction with a binomial random variable with prior prediction probability'''

    delta_gap_time = [7, ]
    delta_prev_time_start = [8, 15, 21, 28, 35]

    for feat in trainDf.columns:
        if feat == 'date' or feat == 'forums':
            continue

        print('Computing for feature: ', feat)

        trainDf_curr = trainDf[['date', feat]]
        testDf_curr = testDf[['date', feat]]

        datePredict = testOutput.copy()

        for dgt in delta_gap_time:

            for dprev in delta_prev_time_start:

                if dgt >= dprev:
                    continue

                X_train, y_train, = prepareData(trainDf_curr, trainOutput, dgt, dprev)
                X_test, y_test,  = prepareData(testDf_curr, testOutput, dgt, dprev)

                ''' Fit a GLM model with logit link on a binomial distribution data '''
                # TODO: Add sparsity for the single predictor models
                # TODO: FOr combined predictors models, add group lasso sparsity - how to define the groups

                y_train = y_train.flatten()
                y_train = y_train.astype(int)
                y_test = y_test.flatten()
                y_test = y_test.astype(int)

                clf = linear_model.LogisticRegression(penalty='l2', class_weight='balanced')
                # clf = ensemble.RandomForestClassifier()

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                datePredict['pred_' + 'eta_' + str(dgt) + '_delta_' + str(dprev)] = y_pred

        pickle.dump(datePredict, open('../../data/results/01_25/regular/malicious_email/LR_L2/'
                                      + str(feat) + '_predictDict.pickle', 'wb'))


if __name__ == "__main__":
    main()
