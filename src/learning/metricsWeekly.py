import pandas as pd
import pickle
import sklearn.metrics
import numpy as np
import datetime

class ArgsStruct:
    LOAD_DATA = True


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


def anomalyDetectWeeks(featPredDf, weeks):

    featPredDf['date'] = pd.to_datetime(featPredDf['date'])
    tprList = []
    fprList = []
    for tf in featPredDf.columns:
        if tf == 'actual' or tf == 'date':
            continue

        y_act = []
        y_pred = []

        for st, end in weeks:
            currDf = featPredDf[featPredDf['date'] >= pd.to_datetime(st)]
            currDf = currDf[currDf['date'] < pd.to_datetime(end)]

            y_pred.extend(currDf[tf])
            y_act.extend(currDf['actual'])

        tpr, fpr = roc_metrics(np.array(y_act), np.array(y_pred))

        tprList.append(tpr)
        fprList.append(fpr)

    return tprList, fprList


def classiicationWeeks(featPredDf, weeks, featStr):
    '''

    :param featPredDf:
    :param weeks: The set of weeks for which the precision and recall has to be measured
    :param featStr: This contains the time parameters and th e feature information
    :return:
    '''

    y_act = []
    y_pred = []

    featPredDf['date'] = pd.to_datetime(featPredDf['date'])
    for st, end in weeks:
        pred_Curr = featPredDf[featPredDf['date'] >= pd.to_datetime(st)]
        pred_Curr = pred_Curr[pred_Curr['date'] < pd.to_datetime(end)]

        y_act.extend(pred_Curr['attackFlag'])
        y_pred.extend(pred_Curr[featStr])

    prec, rec, f1_score = sklearn.metrics.precision_score(y_act, y_pred), \
                          sklearn.metrics.recall_score(y_act, y_pred), \
                          sklearn.metrics.f1_score(y_act, y_pred),

    f1_rand = 0
    for idx_rand in range(5):
        # y_random = np.random.randint(2, size=len(y_act))
        y_random = np.random.binomial(1, 0.26, len(y_act))
        # print(rand)
        f1_rand += sklearn.metrics.f1_score(y_act, y_random)
    f1_rand /= 5

    print("Feature + Time: ", featStr)
    print("Precision %f, Recall %f, F1 %f, Random F1 %f, " %(prec, rec, f1_score, f1_rand))

    return prec, rec, f1_score


# FIrst select the few attack days of each event depending on the count of attacks
def select_attack_days(eventsDf, start_date, thresh):
    '''

    :param eventsDf:
    :param thresh: threshold number of attacks for significance
    :return:
    '''

    weeks = []
    for idx, row in eventsDf.iterrows():
        if row['number_attacks'] >= thresh and row['start_dates'] > start_date.date():
            weeks.append((row['start_dates'], row['end_dates']))

    return weeks


def main():
    featList = ['expert_NonInteractions', 'communityCount', 'shortestPaths', 'CondExperts',
                'numUsers', 'numVulnerabilities', 'numThreads', 'expertsThreads']

    # Weekly occurrence of cve attacks - the CVEs are attached
    cve_eventsDf = pd.read_pickle('../../data/DW_data/CPE_events_corr_me.pickle')

    start_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')
    ''' Select the important weeks from all the weeks of attacks - attack-type wise'''
    weeks_imp = select_attack_days(cve_eventsDf, start_date, 5)

    delta_gap_time = [7, ]
    delta_prev_time_start = [8,]

    metricsDict = {}
    for feat in featList:
        if feat == 'date' or feat == 'forums':
            continue

        metricsDict[feat] = {}
        print('Computing for feature: ', feat)

        feat_predictDf = pickle.load(open('../../data/results/01_25/supervised/malicious_email/LR_L2/featPredict_Df/'
                                          + str(feat) + '_predictDict.pickle', 'rb'))

        anom_predictDf = pickle.load(open('../../data/results/01_25/anomaly/malicious_email/thresh_anom_1/weekly_best/'
                                      + str(feat) + '_predictDict.pickle', 'rb'))

        ''' For supervised prediction '''
        # for dgt in delta_gap_time:
        #     for dprev in delta_prev_time_start:
        #
        #         if dgt >= dprev:
        #             continue
        #
        #         featString = 'pred_' + 'eta_' + str(dgt) + '_delta_' + str(dprev)
        #
        #         prec, rec, f1 = classiicationWeeks(feat_predictDf, weeks_imp, featString)
        #
        #         metricsDict[feat]['prec'] = prec
        #         metricsDict[feat]['rec'] = rec
        #         metricsDict[feat]['f1'] = f1

        ''' For anomaly detetor based prediction '''
        tprList, fprList = anomalyDetectWeeks(anom_predictDf, weeks_imp)
        metricsDict[feat]['tprList'] = tprList
        metricsDict[feat]['fprList'] = fprList

    pickle.dump(metricsDict, open('../../data/results/01_25/anomaly/malicious_email/metricsWeekly_best.pickle', 'wb') )

if __name__ == "__main__":
    main()
