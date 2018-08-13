import sys
sys.path.append('../test')
sys.path.append('../lib')
from pyglmnet import GLM
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn import linear_model, ensemble
import sklearn.metrics


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

    y_actual = outputDf['attackFlag']

    test_start_date = outputDf.iloc[0, 0]
    test_end_date = outputDf.iloc[-1, 0]

    week_predDf = pd.DataFrame()
    count_t = 0
    ''' Iterate over all the thresholds for ROC curves '''
    for t in thresholds_anom:
        # print('Threshold: ', t)
        currDate = test_start_date
        y_estimate = np.zeros(y_actual.shape)
        countDayIndx = 0
        dateList = []
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

        week_predDf['pred_thresh_' + str(count_t)] = y_estimate
        count_t += 1

    week_predDf['actual'] = y_actual
    week_predDf['date'] = dateList

    return week_predDf


def main():
    args = ArgsStruct()
    args.LOAD_DATA = True
    args.forumsSplit = False
    args.FEAT_TYPE = "SNA" # REGULAR: graph features, ANOM: anomaly features, SNA: for social network features
    args.TYPE_PLOT = 'LINE'

    ''' SET THE TRAINING AND TEST TIME PERIODS - THIS IS MANUAL '''
    trainStart_date = datetime.datetime.strptime('2017-01-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-03-16', '%Y-%m-%d')

    testStart_date = datetime.datetime.strptime('2017-03-16', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-07-16', '%Y-%m-%d')

    if args.LOAD_DATA:
        amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
        amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

        if args.FEAT_TYPE == "REGULAR":
            feat_df = pickle.load(open('../../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle', 'rb'))
        elif args.FEAT_TYPE == "ANOM":
            feat_df = pickle.load(
                open('../../data/DW_data/features/feat_combine/user_interStats_Delta_T0_Sep16-Aug17.pickle', 'rb'))
        else:
            feat_df = pickle.load(
                open('../../data/DW_data/features/feat_combine/SNA_Mar16-Apr17_TP50.pickle', 'rb'))

        # Prepare the dataframe for output
        trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)

        # the previous month is needed for features - since we measure lag correlation for the prediction
        instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features

        # Make the date common for all
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        trainDf = feat_df[feat_df['date'] >= instance_TrainStartDate]
        trainDf = trainDf[trainDf['date'] < trainEnd_date]

        instance_TestStartDate = testStart_date - relativedelta(months=1)
        testDf = feat_df[feat_df['date'] >= instance_TestStartDate]
        testDf = testDf[testDf['date'] < testEnd_date]
        testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)

    delta_gap_time = [7, ]
    delta_prev_time_start = [8, ]
    thresh_anom_count = 1

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


                '''
                1. Supervised Classification
                '''
                clf = linear_model.LogisticRegression(penalty='l2', class_weight='balanced')
                # clf = ensemble.RandomForestClassifier()

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                datePredict['pred_' + 'eta_' + str(dgt) + '_delta_' + str(dprev)] = y_pred

                '''
                2. Unsupervised classification: Anomaly detection
                '''
                # thresh_min = 0.1 * np.mean(X_test)
                # thresh_max = 10 * np.mean(np.array(X_test))
                #
                # thresh_anomList = np.arange(thresh_min, thresh_max, (thresh_max - thresh_min) / 50)
                #
                # datePredict = predictAttacks_onAnomaly(testDf, testOutput, dgt, dprev,
                #                                              thresh_anomList, feat, thresh_anom_count)


        pickle.dump(datePredict, open('../../data/results/05_23/SNA/malicious_email/'
                                      + str(feat) + '_predictDict.pickle', 'wb'))
        # pickle.dump(datePredict, open('../../data/results/05_05/unsupervised/malicious_email/cve_0199/'
        #                               + str(feat) + '_predictDict.pickle', 'wb'))


if __name__ == "__main__":
    '''
    The following is for manually evaulating certain weeks based on CVE discussions
    '''
    # dict_df = pd.read_pickle('../../data/results/05_05/unsupervised/malicious_email/cve_0199/'
    #                                   + 'CondExperts' + '_predictDict.pickle')
    #
    # y_act = np.array(dict_df['actual'].tolist())
    # # print(y_act)
    # # print(y_act[y_act == 1.].shape)
    # y_rand = np.random.randint(2, size=len(y_act))
    # for col in dict_df.columns:
    #     if col =="date" or col == "actual":
    #         continue
    #
    #     y_pred = np.array(dict_df[col].tolist())
    #     prec, rec, f1_score = sklearn.metrics.precision_score(y_act, y_pred), \
    #                               sklearn.metrics.recall_score(y_act, y_pred), \
    #                               sklearn.metrics.f1_score(y_act, y_pred)
    #
    #     prec_r, rec_r, f1_score_r = sklearn.metrics.precision_score(y_act, y_rand), \
    #                           sklearn.metrics.recall_score(y_act, y_rand), \
    #                           sklearn.metrics.f1_score(y_act, y_rand)
    #
    #     print(col, prec, rec, f1_score, prec_r, f1_score_r)

    main()
