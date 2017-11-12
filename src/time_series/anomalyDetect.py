import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

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
        num_postsList = list(currTS[feat])
        for idx_t in range(len(currTS)):
            A[idx_t, idx_f] = num_postsList[idx_t]

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
    # return comp

    pca_var = pca.explained_variance_ratio_

    return comp


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


def projectSubspace(components, data):
    for c in range(components.shape[1]): # the column vector corresponds to each component
        # print(data.shape, components[:, c].shape)
        pAxis_vector = np.dot(data, components[c, :])
        pA_norm = np.linalg.norm(pAxis_vector)
        pAxis_vector /= pA_norm


    # Set the threshold to choose the normal subspaces - for now choose the first 6 subspaces
    # as normal and the next 10 as anomalous

    normal_subspace = components[:, :6]
    anomaly_subspace = components[:, 6: 16]

    return normal_subspace, anomaly_subspace


def projectionSeparation(normal_subspace, anomaly_subspace, data):
    PP_T_normal = np.dot(normal_subspace, np.transpose(normal_subspace))
    PP_T_anomaly = np.dot(anomaly_subspace, np.transpose(anomaly_subspace))

    state_vec = np.dot(PP_T_normal, data)
    residual_vec = np.dot(PP_T_anomaly, data)

    return state_vec, residual_vec


def formProjectedTS(normal_sub, anomaly_sub, test_data):
    y_res = np.zeros((test_data.shape[0], 1))
    y_state = np.zeros((test_data.shape[0], 1))
    for idx_t in range(test_data.shape[0]):
        sVEc, rVec = projectionSeparation(normal_sub, anomaly_sub, test_data[idx_t, :])
        y_state[idx_t] = np.linalg.norm(sVEc)
        y_res[idx_t] = np.linalg.norm(rVec)

    return y_state, y_res


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

    # For now, just consider the  
    currDay = start_date
    while(currDay < end_date):
        events = eventsDf[eventsDf['date'] == ]

def main():
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']
    prepareOutput(amEvents_malware)

    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-04-01', '%Y-%m-%d')

    testStart_date = datetime.datetime.strptime('2017-04-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')

    feat_df = pickle.load(open('../../data/DW_data/feature_df_Sept16-Apr17.pickle', 'rb'))
    # feat_df <

    trainDf = feat_df[feat_df['date'] >=trainStart_date.date()]
    trainDf = trainDf[trainDf['date'] < trainEnd_date.date()]

    testDf = feat_df[feat_df['date'] >= testStart_date.date()]
    testDf = testDf[testDf['date'] < testEnd_date.date()]






if __name__ == "__main__":
    main()

