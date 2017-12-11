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

def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def formTSMatrix(forumTS, feat):
    """

    :param forumTS:
    :return: A  - matrix of time series t x f containing number of posts
                    t - time
                    f - forum
    """
    forums = list(set(forumTS['forum']))
    len_TS = len(forumTS[forumTS['forum'] == forums[0]])
    A = np.zeros((len_TS, len(forums)))

    for idx_f in range(len(forums)):
        f = forums[idx_f]
        currTS = forumTS[forumTS['forum'] == f]
        featList = list(currTS[feat])
        for idx_t in range(len(currTS)):
            A[idx_t, idx_f] = featList[idx_t]

    centerd_A = np.zeros(A.shape)
    for c in range(centerd_A.shape[1]):
        colMean = np.mean(A[:, c])
        for r in range(centerd_A.shape[0]):
            centerd_A[r, c] = A[r, c] - colMean

    return A, centerd_A


def getTopComponents(forumMat, numComp):
    # print(forumMat.shape)
    pca = PCA(n_components=numComp)
    pca.fit(forumMat)

    comp = np.transpose(pca.components_)

    pca_var = pca.explained_variance_ratio_

    return comp, pca_var


    # for c in range(comp.shape[1]):
    #     data = comp[:, c]
    #
    #     plt.plot(data)
    #     # plt.title('Forum: ' + str(f))
    #     plt.grid()
    #     plt.xticks(size=15)
    #     plt.yticks(size=15)
    #     plt.xlabel('Date Time frame', size=15)
    #     plt.ylabel('Number of conversations', size=15)
    #     plt.subplots_adjust(left=0.17, bottom=0.17, top=0.9)
    #
    #     plt.show()
    #     manager = plt.get_current_fig_manager()
    #     manager.resize(*manager.window.maxsize())
    #
    #     # plt.savefig('../../plots/dw_stats/forums_postTS/forum_' + str(f) + '.png' )
    #     plt.close()


def projectSubspace(components, data, num_normal):
    # for c in range(components.shape[1]): # the column vector corresponds to each component
    #     # print(data.shape, components[:, c].shape)
    #     pAxis_vector = np.dot(data, components[c, :])
    #     pA_norm = np.linalg.norm(pAxis_vector)
    #     pAxis_vector /= pA_norm

    normal_subspace = components[:, :num_normal]
    anomaly_subspace = components[:, num_normal:]

    # print(anomaly_subspace)
    return normal_subspace, anomaly_subspace


def projectionSeparation(normal_subspace, anomaly_subspace, data):
    PP_T_normal = np.dot(normal_subspace, np.transpose(normal_subspace))
    PP_T_anomaly = np.dot(anomaly_subspace, np.transpose(anomaly_subspace))

    # print(PP_T_anomaly)
    state_vec = np.dot(PP_T_normal, data.transpose())
    residual_vec = np.dot(PP_T_anomaly, data.transpose())

    return state_vec, residual_vec


def formProjectedTS(normal_sub, anomaly_sub, test_data):
    y_res = np.zeros((test_data.shape[0], 1))
    y_state = np.zeros((test_data.shape[0], 1))
    for idx_t in range(test_data.shape[0]):
        sVEc, rVec = projectionSeparation(normal_sub, anomaly_sub, test_data[idx_t, :])
        y_state[idx_t] = np.linalg.norm(sVEc)
        y_res[idx_t] = np.linalg.norm(rVec)

    return y_state, y_res


def Q_statistic(top_comp, num_norm_comp, data):
    phi = [0. for _ in range(3)]
    for c in range(num_norm_comp, top_comp.shape[1]):
        variance = np.power(np.linalg.norm(np.dot(data, top_comp[:, c])), 2)
        phi[0] += variance
        phi[1] += (np.power(variance, 2))
        phi[2] += (np.power(variance, 3))

    h_0 = 1 - (2*phi[0]*phi[2]/(3*np.power(phi[1], 2)))
    c_alpha = 2.58

    numer = (c_alpha*np.power(2*phi[1] * np.power(h_0, 1), 0.5) / phi[0]) + 1 \
            + (phi[1]*h_0*(h_0-1)/np.power(phi[0], 2))

    q_value = phi[0] * np.power(numer, (1/h_0))

    return q_value


def plot_ROC(normal_sub, anomaly_sub, test_data, amEventsDates, feat):
    perc = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # perc = [1]
    test_mat, cent_test_mat = formTSMatrix(test_data, feat)

    stateVec, resVec = formProjectedTS(normal_sub, anomaly_sub, test_mat)
    test_dates = list(test_data['date'])

    dates_consider = []
    for d in test_dates:
        dates_consider.append(str(d)[:10])

    max_val = np.max(resVec)
    TPR = []
    FPR = []
    for p in perc:
        thresh = p * max_val
        anomalies_list = []
        for idx in  range(len(resVec)):
            if resVec[idx] >= thresh:
                anomalies_list.append(dates_consider[idx])

        date_pred = []
        for idx_anom in range(len(anomalies_list)):
            date_anom = anomalies_list[idx_anom]

            date_anom = datetime.datetime.strptime(date_anom, "%Y-%m-%d")
            date_pred_1 = str(date_anom + datetime.timedelta(days=7))[:10]
            date_pred_2 = str(date_anom + datetime.timedelta(days=8))[:10]
            date_pred_3 = str(date_anom + datetime.timedelta(days = 9))[:10]

            date_pred.append(date_pred_1)
            date_pred.append(date_pred_2)
            date_pred.append(date_pred_3)

        date_pred = list(set(date_pred))

        tp, fp, tn, fn = 0., 0., 0., 0.
        for d in dates_consider:
            if d in date_pred and d in amEventsDates:
                tp += 1
            elif d in date_pred and d not in amEventsDates:
                fp += 1
            elif d not in date_pred and d in amEventsDates:
                fn += 1
            else:
                tn += 1

        TPR.append(tp/(tp+fn))
        FPR.append(fp/(fp+tn))

    # plt.scatter(TPR, FPR)
    # plt.plot(TPR, FPR)
    # plt.grid()
    # plt.xticks(size=15)
    # plt.yticks(size=15)
    # plt.xlabel('True Positive Rate', size=15)
    # plt.ylabel('False Positive Rate', size=15)
    # plt.subplots_adjust(left=0.17, bottom=0.17, top=0.9)
    #
    # plt.show()

    return TPR, FPR


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

    tpr = 0.
    fpr = 0.
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


def predictAttacks_onAnomaly(input, output, thresholds):
    '''

    :param input:
    :param output:
    :return:
    '''

    ''' Metrics for ROC curve '''
    tprList = []
    fprList = []

    y_actual = output['attackFlag']

    test_start_date = output.iloc[0, 0]
    test_end_date = output.iloc[-1, 0]

    delta_prev_time = 3 # no of days to check before the week of current day
    maxPrec = -10
    maxRec = -10
    f1_score = -10

    for t in thresholds:
        # print('Threshold: ', t)
        currDate = test_start_date
        y_estimate = np.zeros(y_actual.shape)
        countDayIndx = 0
        while(currDate <= test_end_date):
            '''This loop checks values on either of delta days prior'''
            for idx in range(delta_prev_time):
                historical_day = currDate - datetime.timedelta(days=(7-idx))
                res_vec_value = input[input['date'] == historical_day]['res_vec']

                try:
                    if res_vec_value.iloc[0] > t:
                        y_estimate[countDayIndx] = 1
                        break
                except:
                    # print(currDate)
                    continue
            countDayIndx += 1
            currDate += datetime.timedelta(days=1)

        tpr, fpr = roc_metrics(y_actual, y_estimate)
        tprList.append(tpr)
        fprList.append(fpr)

        if maxPrec < sklearn.metrics.precision_score(y_actual, y_estimate):
            maxPrec = sklearn.metrics.precision_score(y_actual, y_estimate)
            maxRec = sklearn.metrics.recall_score(y_actual, y_estimate)
            f1_score = sklearn.metrics.f1_score(y_actual, y_estimate)

    return maxPrec, maxRec, f1_score

    # print(tprList)
    # print(fprList)
    # plt.plot(tprList, fprList, color='black', linewidth=3)
    # # plt.plot(TPR, FPR)
    # plt.grid()
    # plt.xticks(size=15)
    # plt.yticks(size=15)
    # plt.xlabel('True Positive Rate', size=15)
    # plt.ylabel('False Positive Rate', size=15)
    # plt.subplots_adjust(left=0.17, bottom=0.17, top=0.9)
    #
    # plt.show()


def anomalyVec(res_vec, ):
    mean_rvec = np.mean(res_vec)

    anomaly_vec = np.zeros(res_vec.shape)
    for t in range(res_vec.shape[0]):
        if res_vec[t] > mean_rvec:
            anomaly_vec[t] = 1.

    return anomaly_vec


def prepareData(inputDf, outputDf):
    y_actual = outputDf['attackFlag']
    w_1 = 0.5
    w_2 = 0.5

    # print(inputDf)
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
                feat_vec = inputDf[inputDf['date'] == historical_day]
                # X[countDayIndx, idx] = (0.5 * feat_vec['state_vec'].iloc[0]) + (0.5 * feat_vec['res_vec'].iloc[0])

                if feat_vec['anomaly_vec'].iloc[0] == 1:
                    X[countDayIndx, idx] = (feat_vec['state_vec'].iloc[0]) + (feat_vec['res_vec'].iloc[0])
                else:
                    X[countDayIndx, idx] = (feat_vec['state_vec'].iloc[0]) + (feat_vec['res_vec'].iloc[0])
            except:
                continue
        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    return X, y_actual


def trainModel(trainDf, featStr, forums):
    forumTSMat, forumTSMAtCent = formTSMatrix(trainDf, featStr)
    # print(forumTSMat.shape)

    ''' Get the PCA components'''
    num_comp = 8
    top_comp, variance_comp = getTopComponents(forumTSMAtCent, num_comp)
    # print(variance_comp)
    # print(top_comp.shape)

    ''' Find the normal and residual subspace '''
    # keep the first 3 components as normal subspace
    normal_subspace, residual_subspace = projectSubspace(top_comp, forumTSMAtCent, 3)

    """ Check the q-value statistic --- some error !!! """
    # q_value = Q_statistic(top_comp, 3, forumTSMAtCent)

    ''' Compute the separation matrix '''
    state_vec, res_vec = projectionSeparation(normal_subspace, residual_subspace, forumTSMAtCent)
    featValue = (np.power(np.linalg.norm(state_vec, axis=0), 2)).tolist()
    minVal = min(featValue)
    meanVal = 3 * np.mean(np.array(featValue))
    numThresholds = 40
    partitionRange = (meanVal - minVal) / numThresholds
    thresh = []

    for i in range(numThresholds - 1, 0, -1):
        thresh.append(minVal + (i * partitionRange))

    df_train_feat = pd.DataFrame()
    df_train_feat['date'] = trainDf[trainDf['forum'] == forums[0]]['date']
    df_train_feat[featStr + '_state_vec'] = np.power(np.linalg.norm(state_vec, axis=0), 2)
    # df_train_feat[featStr + '_res_vec'] = np.power(np.linalg.norm(res_vec, axis=0), 2)

    # anomaly_vector_train = anomalyVec(np.array(df_train_feat['res_vec']))
    #
    # df_train_feat['anomaly_vec'] = anomaly_vector_train
    #
    # ''' The testing procedure begins'''
    #
    # forumTSMat_test, forumTSMAtCent_test = formTSMatrix(testDf, feat)
    # state_vec_test, res_vec_test = projectionSeparation(normal_subspace, residual_subspace, forumTSMAtCent_test)
    #
    # df_test_feat = pd.DataFrame()
    # df_test_feat['date'] = testDf[testDf['forum'] == forums[0]]['date']
    # df_test_feat['state_vec'] = np.power(np.linalg.norm(state_vec_test, axis=0), 2)
    # df_test_feat['res_vec'] = np.power(np.linalg.norm(res_vec_test, axis=0), 2)
    #
    # anomaly_vector_test = anomalyVec(np.array(df_test_feat['res_vec']))
    #
    # df_test_feat['anomaly_vec'] = anomaly_vector_test

    # df_train_feat.plot(figsize=(12,8), x='date', y='state_vec', color='black', linewidth=2)
    # plt.grid(True)
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    # plt.xlabel('date', size=20)
    # plt.ylabel(feat, size=20)
    # plt.title('Residual Vector')
    # plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    # plt.show()
    # plt.close()

    # X_train, y_train = prepareData(df_train_feat, trainOutput)
    # # print(df_train_feat)
    # logreg = linear_model.LogisticRegression(penalty='l2')
    # # logreg = ensemble.RandomForestClassifier()
    # logreg.fit(X_train, y_train)
    #
    # X_test, y_test = prepareData(df_test_feat, testOutput)
    # # print(list(y_test))
    # y_pred = logreg.predict(X_test)
    #
    # # Attack prediction evaluation
    # prec, rec, f1_score = sklearn.metrics.precision_score(y_actual_test, y_pred), \
    #                       sklearn.metrics.recall_score(y_actual_test, y_pred), \
    #                       sklearn.metrics.f1_score(y_actual_test, y_pred),
    #
    # print(feat, prec, rec, f1_score)
    #
    # # Attack prediction evaluation
    # prec, rec, f1_score = predictAttacks_onAnomaly(df_test_feat, testOutput, thresh)
    # print(feat, prec, rec, f1_score)
    return df_train_feat

def main():
    forums = [35, 38, 60, 62, 71, 84, 88, 105, 133, 135, 146, 147, 150, 161, 173, 179, 197, ]
    # forums = [35,]

    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    trainStart_date = datetime.datetime.strptime('2016-9-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')


    #
    # trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
    #
    # testStart_date = datetime.datetime.strptime('2017-02-01', '%Y-%m-%d')
    # testEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')
    #
    # testDf = feat_df[feat_df['date'] >= testStart_date.date()]
    # testDf = testDf[testDf['date'] < testEnd_date.date()]
    #
    # '''   THIS TIMEFRAME IS IMPORTANT  !!!! '''
    # testOutput = prepareOutput(amEvents_malware, testStart_date + relativedelta(months=1),
    #                            testEnd_date)
    #
    # y_actual_test = list(testOutput['attackFlag'])
    # y_random = y_actual_test.copy()
    #
    # shuffle(y_random)
    # random_prec = sklearn.metrics.precision_score(y_actual_test, y_random)
    # random_recall = sklearn.metrics.recall_score(y_actual_test, y_random)
    # random_f1 = sklearn.metrics.f1_score(y_actual_test, y_random)
    # print('Random: ', random_prec, random_recall, random_f1)


    ''' Concatenat the features into a single dataframe'''
    fileName_prefix = ['shortestPaths', 'conductance', 'commThreads']
    feat_df = pd.DataFrame()
    for fp in fileName_prefix:
        if feat_df.empty:
            feat_df = pd.read_pickle('../../data/DW_data/features/' + str(fp) + '_DeltaT_4_Sept16-Apr17_TP10.pickle')
        else:
            curr_df = pd.read_pickle('../../data/DW_data/features/' + str(fp) + '_DeltaT_4_Sept16-Apr17_TP10.pickle')
            feat_df = pd.merge(feat_df, curr_df, on=['date', 'forum'])

    feat_df = feat_df[feat_df['forum'].isin(forums)]
    trainDf = feat_df[feat_df['date'] >= trainStart_date]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]

    ''' Plot the features forum wise '''
    features = ['shortestPaths', 'CondExperts', 'expertsThreads']

    for feat in features:
        for f in forums:
            plotDf_forum = trainDf[trainDf['forum'] == f]
            for idx_cpe in range(10):
                dir_save = '../../plots/dw_stats/feat_plot/' + str(feat) + '/'
                if not os.path.exists(dir_save):
                    os.makedirs(dir_save)
                featStr = feat+'_CPE_R' + str(idx_cpe+1)
                plotDf_forumCPE = plotDf_forum[(plotDf_forum[featStr] != -1.) & (plotDf_forum[featStr] != 0.)][featStr]
                median_value = np.median(np.array(list(plotDf_forumCPE)))
                if np.isnan(median_value):
                    median_value = 0.
                trainDf.ix[trainDf.forum==f, featStr] = trainDf[trainDf['forum'] == f][featStr].replace([-1.00, 0.0, 0.00], median_value)
                # plotDf_forumCPE = trainDf[trainDf['forum'] == f]
                #
                # # print(plotDf_forumCPE[featStr])
                # plotDf_forumCPE.plot(figsize=(12,8), x='date', y=featStr, color='black', linewidth=2)
                # plt.grid(True)
                # plt.xticks(size=20)
                # plt.yticks(size=20)
                # plt.xlabel('date', size=20)
                # plt.ylabel(feat, size=20)
                # plt.title('Forum=' + str(f))
                # plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
                # # plt.show()
                # file_save = dir_save + 'forum_' + str(f) +  '_CPE_R' + str(idx_cpe+1)
                # plt.savefig(file_save)
                # plt.close()

    ''' Form the residual and normal time series for each feaure, for each CPE'''
    subspace_df = pd.DataFrame()
    for feat in features:
        for idx_cpe in range(5):
            featStr = feat + '_CPE_R' + str(idx_cpe + 1)
            trainDf_subspace  = trainModel(trainDf, featStr, forums)
            if subspace_df.empty:
                subspace_df = trainDf_subspace
            else:
                subspace_df = pd.merge(subspace_df, trainDf_subspace, on=['date'])

    pickle.dump(subspace_df, open('../../data/DW_data/features/subspace_df_v12_10.pickle', 'wb'))

    # print(subspace_df)
    # for column in subspace_df:
    #     if column == 'date':
    #         continue
    #     dir_save = '../../plots/dw_stats/subspace_plot/'
    #     if not os.path.exists(dir_save):
    #         os.makedirs(dir_save)
    #
    #     subspace_df.plot(figsize=(12,8), x='date', y=column, color='black', linewidth=2)
    #     plt.grid(True)
    #     plt.xticks(size=20)
    #     plt.yticks(size=20)
    #     plt.xlabel('date', size=20)
    #     plt.ylabel(column, size=20)
    #     # plt.title('Forum=' + str(f))
    #     plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
    #     # plt.show()
    #     file_save = dir_save + column
    #     plt.savefig(file_save)
    #     plt.close()

if __name__ == "__main__":
    main()

