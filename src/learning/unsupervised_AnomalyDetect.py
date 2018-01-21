import sys
sys.path.append('../test')
sys.path.append('../lib')
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import sklearn.metrics
from random import shuffle


class ArgsStruct:
    LOAD_DATA = True


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


def roc_metrics(y_actual, y_estimate):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(y_actual.shape[0]):
        if y_actual[i] == y_estimate[i] and y_actual[i] == 1:
            tp += 1
        elif y_actual[i] == y_estimate[i] and y_actual[i] == 0.:
            tn += 1
        elif y_actual[i] != y_estimate[i] and y_actual[i] == 0.:
            fp += 1
        else:
            fn += 1

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return tpr, fpr


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
            if count_anomalies >= (delta_gap_time/7 * thresh_anom_count):
                y_estimate[countDayIndx] = 1

            countDayIndx += 1
            currDate += datetime.timedelta(days=1)

        tpr, fpr = roc_metrics(y_actual, y_estimate)
        tprList.append(tpr)
        fprList.append(fpr)


    return tprList, fprList,

def main():
    args = ArgsStruct()
    args.LOAD_DATA = True
    args.forumsSplit = False

    ''' SET THE TRAINING AND TEST TIME PERIODS - THIS IS MANUAL '''
    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')

    testStart_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    if args.LOAD_DATA:
        amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
        amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

        feat_df = pickle.load(open('../../data/DW_data/features/feat_forums/subspace_df_v01_05.pickle', 'rb'))

        feat_df['date'] = pd.to_datetime(feat_df['date'])

        # Training dataframe
        trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
        # the previous month is needed for features - since we measure lag correlation for the prediction
        instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features
        trainDf = feat_df[feat_df['date'] >= instance_TrainStartDate]
        trainDf = trainDf[trainDf['date'] < trainEnd_date]

        #Testing dataframe
        testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)
        instance_TestStartDate = testStart_date - relativedelta(months=1)
        testDf = feat_df[feat_df['date'] >= instance_TestStartDate]
        testDf = testDf[testDf['date'] < testEnd_date]


    # parameters for model
    delta_gap_time = [7, 14]
    delta_prev_time_start = [8, 14, 21, 28, 35]
    thresh_anom_count = 1

    rocList = {}
    for dgt in delta_gap_time:
        for dprev in delta_prev_time_start:

            if dgt >= dprev:
                continue
            for feat in trainDf.columns:

                if feat == 'date' or feat == 'forums':
                    continue
                rocList[feat] = {}
                if 'state' in feat:
                    continue
                print('Computing for feature: ', feat)

                feat_val = trainDf[feat]
                thresh_min = np.mean(np.array(feat_val))
                thresh_max = 4*np.mean(np.array(feat_val))

                # print(thrsh_min, thresh_max)

                thresh_anomList = np.arange(thresh_min, thresh_max, (thresh_max - thresh_min)/10)


                tprList, fprList, = predictAttacks_onAnomaly(testDf, testOutput, dgt, dprev,
                                                                                       thresh_anomList, feat, thresh_anom_count)

                rocList[feat]['tprList'] = tprList
                rocList[feat]['fprList'] = fprList


            pickle.dump(rocList, open('../../data/results/01_09/anomaly/' + str('res_') + 'tgap_' + str(dgt) + '_tstart_' + str(dprev) + '.pickle', 'wb'))





if __name__ == "__main__":
    main()

