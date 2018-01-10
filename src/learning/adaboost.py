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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn import preprocessing

class ArgsStruct:
    LOAD_DATA = True

''' The following functions are for formatting the output in a numpy design
    matrix so that normal matrix computations are easier
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
    train_start_date = pd.to_datetime(outputDf.iloc[0, 0]) # first row
    train_end_date = pd.to_datetime(outputDf.iloc[-1, 0]) # last row

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    y_true = outputDf['attackFlag']

    # This can be validated through
    # delta_gap_time = 7  # no of days to check
    # delta_prev_time_start = 28 # no of days to check before the current day

    currDate = train_start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_true.shape[0], delta_gap_time*num_features)) #input
    Y = -np.ones((y_true.shape[0], 1)) #output

    while (currDate <= train_end_date):
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

    return X, Y



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

        if args.FEAT_TYPE == "REGULAR":
            feat_df_reg = pickle.load(open('../../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle', 'rb'))
            feat_df_subs = pickle.load(
                open('../../data/DW_data/features/feat_forums/subspace_df_v01_05.pickle', 'rb'))

            print(feat_df_reg)
            print(feat_df_subs)

        # the previous month is needed for features - since we measure lag correlation for the prediction
        instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features

        # Make the date common for all
        feat_df_reg['date'] = pd.to_datetime(feat_df_reg['date'])
        trainDf_reg = feat_df_reg[feat_df_reg['date'] >= instance_TrainStartDate]
        trainDf_reg = trainDf_reg[trainDf_reg['date'] < trainEnd_date]

        feat_df_subs['date'] = pd.to_datetime(feat_df_subs['date'])
        trainDf_subs = feat_df_subs[feat_df_subs['date'] >= instance_TrainStartDate]
        trainDf_subs = trainDf_subs[trainDf_subs['date'] < trainEnd_date]

        # Prepare the dataframe for output
        trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)

        instance_TestStartDate = testStart_date - relativedelta(months=1)
        testDf_reg = feat_df_reg[feat_df_reg['date'] >= instance_TestStartDate]
        testDf_reg = testDf_reg[testDf_reg['date'] < testEnd_date]

        testDf_subs = feat_df_subs[feat_df_subs['date'] >= instance_TestStartDate]
        testDf_subs = testDf_subs[testDf_subs['date'] < testEnd_date]

        testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)
        y_actual_test = list(testOutput['attackFlag'])

        y_random = np.random.randint(2, size=len(y_actual_test))
        random_prec = sklearn.metrics.precision_score(y_actual_test, y_random)
        random_recall = sklearn.metrics.recall_score(y_actual_test, y_random)
        random_f1 = sklearn.metrics.f1_score(y_actual_test, y_random)
        print('Random: ', random_prec, random_recall, random_f1)

    delta_gap_time = [7, 10, 14, 17, 21]
    delta_prev_time_start = [8, 14, 21, 28]


    for dgt in delta_gap_time:
        for dprev in delta_prev_time_start:
            if dgt >= dprev:
                continue

            outputDf = pd.DataFrame()
            for feat in trainDf_reg.columns:
                if feat == 'date' or feat == 'forums':
                    continue

                print('Computing for feature: ', feat)

                trainDf_curr_reg = trainDf_reg[['date', feat]]
                testDf_curr_reg = testDf_reg[['date', feat]]

                try:
                    trainDf_curr_subs = trainDf_subs[['date', feat + '_res_vec']]
                    testDf_curr_subs = testDf_subs[['date', feat + '_res_vec']]
                except KeyError:
                    continue

                trainDf_curr = pd.merge(trainDf_curr_reg, trainDf_curr_subs, on=['date'])
                testDf_curr = pd.merge(testDf_curr_reg, testDf_curr_subs, on=['date'])

                X_train, y_train = prepareData(trainDf_curr, trainOutput, dgt, dprev)
                X_test, y_test = prepareData(testDf_curr, testOutput, dgt, dprev)

                y_train = y_train.flatten()
                y_train = y_train.astype(int)
                y_test = y_test.flatten()
                y_test = y_test.astype(int)

                # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                #                  algorithm="SAMME",
                #                  n_estimators=50)
                clf = linear_model.LogisticRegression(penalty='l2', class_weight='balanced')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)


                prec, rec, f1_score = sklearn.metrics.precision_score(y_test, y_pred),\
                sklearn.metrics.recall_score(y_test, y_pred), \
                sklearn.metrics.f1_score(y_test, y_pred),

                outputDf[feat] = [prec, rec, f1_score]

                # print('Random: ', random_prec, random_recall, random_f1)
                # print(prec, rec, f1_score)

            meta_data = pd.Series([('Random: '), ('precision: ' + str(random_prec),
                                                  ('recall: ' + str(random_recall)), ('f1: ' + str(random_f1)))])
            with open('../../data/results/01_09/anomaly/LR_L2/' + str('regular_') + 'tgap_' + str(dgt) + '_tstart_' + str(
                    dprev) + '.csv', 'w') as fout:
                fout.write('meta data\n:')
                meta_data.to_csv(fout)
                outputDf.to_csv(fout)

if __name__ == "__main__":
    main()
