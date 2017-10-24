import sys
sys.path.insert(0, '../network_analysis/')
sys.path.insert(0, '../load_data/')
sys.path.insert(0, '../stat_analysis/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle
import userAnalysis as usAn
import numpy as np
import networkx.algorithms.cuts as nxCut
import datetime
import createConnections as ccon
import load_dataDW as ldDW
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


def user_CVE_groups(cve_cpe_data, vul_data):
    users_CVEMap = {}
    CVE_usersMap = {}
    for cve in cve_cpe_data['cve']:
        vulnItems = vul_data[vul_data['vulnId'] == cve]
        users = vulnItems['users'].tolist()
        if cve not in CVE_usersMap:
            CVE_usersMap[cve] = []
        CVE_usersMap[cve].extend(users)

        if len(users) == 0:
            continue

        usersList = users[0]
        for u in usersList:
            # print(u)
            if u not in users_CVEMap:
                users_CVEMap[u] = []

            users_CVEMap[u].append(cve)

    return users_CVEMap, CVE_usersMap


def getCVEinTopCPE_Groups(start_date, end_date, vulInfo, cveCPE, K):
    """

    :param start_date:
    :param end_date:
    :param vulInfo:
    :param cveCPE:
    :param K:
    :return: CVEs in top CPE groups
    """
    vulCurr = vulInfo[vulInfo['postedDate'] >= start_date]
    vulCurr = vulCurr[vulCurr['postedDate'] < end_date]

    vulnerab = vulCurr['vulnId']
    cveCPE_curr = cveCPE[cveCPE['cve'].isin(vulnerab)]
    topCPEs = {}
    for idx, row in cveCPE_curr.iterrows():
        clTag = row['cluster_tag']
        if clTag not in topCPEs:
            topCPEs[clTag] = 0

        topCPEs[clTag] += 1

    # print(topCPEs)
    if K==-1:
        K = len(topCPEs)
    topCPEs_sorted = sorted(topCPEs.items(), key=operator.itemgetter(1), reverse=True)[:K]
    topCPEsList = []
    for cpe, count in topCPEs_sorted:
        topCPEsList.append(cpe)

    topCVE = cveCPE[cveCPE['cluster_tag'].isin(topCPEsList)]

    return list(topCVE['cve'])


def getExperts(topCVE, CVE_userMap):
    usersCVECount = {}
    for tc in topCVE:
        userCurrCVE = CVE_userMap[tc]
        if len(userCurrCVE) == 0:
            continue
        for u in list(set(userCurrCVE[0])):
            # if u in usersGlobal:
            if u not in usersCVECount:
                usersCVECount[u] = 0
            usersCVECount[u] += 1

    usersSorted = sorted(usersCVECount.items(), key=operator.itemgetter(1), reverse=True)
    mean_count = np.mean(np.array(list(usersCVECount.values())))
    threshold = 0

    usersCVECount_topCPE = {}
    for u, count in usersSorted:
        if count >= threshold:
            usersCVECount_topCPE[u] = count

    return usersCVECount_topCPE


def countConversations(sd, ed, forums):
    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df_postsTS = pd.DataFrame()

    datesList = []
    numPostsList = []
    uidsList = []
    uidsCount = []
    forumsList = []

    for f in forums:
        print("Forum: ", f)
        start_date = sd
        end_date = ed
        start_year = 2016
        start_month = 4
        while start_month <= 12:
            # print("Start Date:", start_date )
            postsDf = ldDW.getDW_data_postgres(forums_list=[f], start_date=start_date, end_date=end_date)
            postsDf['DateTime'] = postsDf['posteddate'].map(str) + ' ' + postsDf['postedtime'].map(str)
            postsDf['DateTime'] = postsDf['DateTime'].apply(lambda x:
                                                            datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            postsDf = postsDf.sort('DateTime', ascending=True)

            start_day = 1
            while True:
                usersTempList = []
                if start_day < 10:
                    start_dayStr = str('0') + str(start_day)
                else:
                    start_dayStr = str(start_day)

                if start_month < 10:
                    start_monthStr = str('0') + str(start_month)
                else:
                    start_monthStr = str(start_month)

                start_date = datetime.datetime.strptime(
                    str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

                end_date = datetime.datetime.strptime(
                    str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 23:59:00',
                    '%Y-%m-%d %H:%M:%S')

                posts_currDay = postsDf[postsDf['DateTime'] >= start_date]
                posts_currDay = posts_currDay[posts_currDay['DateTime'] < end_date]

                datesList.append(start_date)
                numPostsList.append(len(posts_currDay))

                for idx, row in posts_currDay.iterrows():
                    usersTempList.append(row['uid'])

                uidsList.append(usersTempList)
                uidsCount.append(len(list(set(usersTempList))))
                forumsList.append(f)
                start_day += 1

                # Break condition
                if start_day > daysMonths[start_month - 1]:
                    break

            start_month += 1
            if start_month > 12:
                break
            if start_month < 10:
                start_monthStr = str('0') + str(start_month)
            else:
                start_monthStr = str(start_month)
            start_date = str(start_year) + '-' + start_monthStr + '-01'
            end_date = str(start_year) + '-' + start_monthStr + '-' + str(
                                                    daysMonths[start_month - 1])

    df_postsTS['forum'] = forumsList
    df_postsTS['date'] = datesList
    df_postsTS['number_posts'] = numPostsList
    df_postsTS['users'] = uidsList
    df_postsTS['number_users'] = uidsCount

    return df_postsTS


def segmentEventDaily(eventsDf):
    eventsDf = eventsDf.sort('date', ascending=True)

    start_year = eventsDf['date'].iloc[0].year
    start_month = eventsDf['date'].iloc[0].month
    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df_amEventsTS = pd.DataFrame()

    datesList = []
    numEventsList = []
    eventTypeList = []

    while start_month <= 12:
        start_day = 1

        while True:
            eventType = []
            if start_day < 10:
                start_dayStr = str('0') + str(start_day)
            else:
                start_dayStr = str(start_day)

            if start_month < 10:
                start_monthStr = str('0') + str(start_month)
            else:
                start_monthStr = str(start_month)

            start_date = datetime.datetime.strptime(
                str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

            end_date = datetime.datetime.strptime(str(start_year) + '-' + start_monthStr + '-' + start_dayStr + ' 23:59:00',
                                                  '%Y-%m-%d %H:%M:%S')

            events_currDay = eventsDf[eventsDf['date'] >= start_date]
            events_currDay = events_currDay[events_currDay['date'] < end_date]

            datesList.append(start_date)
            numEventsList.append(len(events_currDay))

            for idx, row in events_currDay.iterrows():
                eventType.append(row['event_type'])

            eventTypeList.append(eventType)

            start_day += 1

            # Break condition
            if start_day > daysMonths[start_month-1]:
                break

        start_month += 1
        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)

        # Break condition
        events_nextMonth = eventsDf[eventsDf['date'] >= \
                                  pd.to_datetime(str(start_year)+'-'+start_monthStr+'-01', format='%Y-%m-%d')]
        events_nextMonth = events_nextMonth[events_nextMonth['date'] <= \
                                    pd.to_datetime(str(start_year) + '-' + start_monthStr + '-' + str(daysMonths[start_month-1])
                                                   , format='%Y-%m-%d')]
        if len(events_nextMonth) == 0:
            break

    df_amEventsTS['date'] = datesList
    df_amEventsTS['number_events'] = numEventsList
    df_amEventsTS['event_types'] = eventTypeList

    return df_amEventsTS


def Conductance(network, userG1, userG2):
    conductanceVal = nxCut.conductance(network, userG1, userG2)

    return conductanceVal


def centralities(network, arg, users):
    cent = {}
    if arg == "InDegree":
        cent = nx.in_degree_centrality(network)

    if arg == "OutDegree":
        cent = nx.out_degree_centrality(network)

    if arg == "Pagerank":
        cent = nx.pagerank(network)

    if arg == "core":
        cent = nx.core_number(network)

    cent_sum = 0.
    for u in users:
        cent_sum += cent[u]

    return cent_sum / len(users)


def computeFeatureTimeSeries(start_date, end_date, forums, cve_cpeData, vulnData, postsDailyDf, allPosts):
    KB_gap = 3
    titlesList = []
    feat_topUsers = []
    feat_experts = []
    newUsersWeeklyPercList = []
    conductanceList = []
    prList = []
    datesList = []
    nbrsList = []
    coreList = []
    featDF = pd.DataFrame()

    # Map of users with CVE to user and user to CVE
    users_CVEMap, CVE_usersMap = user_CVE_groups(cve_cpeData, vulnData)

    for idx in range(0, 6, 3):
        # KB network formation
        start_month = int(start_date[5:7]) + idx
        end_month = start_month + KB_gap
        if end_month > 12:
            end_month = 12

        if start_month < 10:
            start_monthStr = str('0') + str(start_month)
        else:
            start_monthStr = str(start_month)

        if end_month < 10:
            end_monthStr = str('0') + str(end_month)
        else:
            end_monthStr = str(end_month)

        start_dateCurr = start_date[:5] + start_monthStr + start_date[7:]
        if end_month == 12:
            end_dateCurr = start_date[:5] + 31 + start_date[7:]
        else:
            end_dateCurr = start_date[:5] + end_monthStr + start_date[7:]
        print("KB info: ")
        print("Start date: ", start_dateCurr, " ,End date: ", end_dateCurr)
        df_KB = allPosts[allPosts['posteddate'] >= start_dateCurr]
        df_KB = df_KB[df_KB['posteddate'] < end_dateCurr]
        # df_KB = ldDW.getDW_data_postgres(forums, start_dateCurr, end_dateCurr)
        threadidsKB = list(set(df_KB['topicid']))
        KB_edges = ccon.storeEdges(df_KB, threadidsKB)

        # Get the users who have mentioned CVEs in their posts
        usersCVE = list(users_CVEMap.keys())

        # networkKB, topUsersKB, usersCVE = splitUsers(users_CVEMap, KB_edges)

        # Find the experts in the KB
        # User Group
        topCVE = getCVEinTopCPE_Groups(start_dateCurr, end_dateCurr, vulData, cve_cpeData, K=20)
        expertsDict = getExperts(topCVE, CVE_usersMap)
        expertUsers = list(expertsDict.keys())

        # Training network formation starts from here
        train_start_month = end_month
        train_end_month = train_start_month + KB_gap
        if train_start_month < 10:
            train_start_monthStr = str('0') + str(train_start_month)
        else:
            train_start_monthStr = str(train_start_month)

        if train_end_month < 10:
            train_end_monthStr = str('0') + str(train_end_month)
        else:
            train_end_monthStr = str(train_end_month)

        train_start_date = start_date[:5] + train_start_monthStr + start_date[7:]
        train_end_date = start_date[:5] + train_end_monthStr + start_date[7:]

        print("Training data info: ")
        print("Start date: ", train_start_date, " ,End_date: ", train_end_date)

        postsDailyDf_curr = postsDailyDf[postsDailyDf['date'] >= train_start_date]
        postsDailyDf_curr = postsDailyDf_curr[postsDailyDf_curr['date'] <= train_end_date]

        # print(postsDailyDf_curr)
        dates_curr = list(postsDailyDf_curr['date'])
        mergeEdges = KB_edges.copy()
        for idx_date in range(len(dates_curr)):
            if idx_date == len(dates_curr)-1:
                break
            datesList.append(dates_curr[idx_date])
            print("Computing for date:", dates_curr[idx_date])
            date_start_curr = str(dates_curr[idx_date])[:10]
            date_end_curr = str(dates_curr[idx_date+1])[:10]

            # print(date_start_curr, date_end_curr)
            df_daily = allPosts[allPosts['posteddate'] >= date_start_curr]
            df_daily = df_daily[df_daily['posteddate'] < date_end_curr]
            # df_daily = ldDW.getDW_data_postgres(forums, date_start_curr, date_end_curr)

            topics_curr = list(set(df_daily['topicid']))
            dailyEdges = ccon.storeEdges(df_daily, topics_curr)

            if len(dailyEdges) == 0:
                conductanceList.append(0)
                continue

            users_curr = list(set(dailyEdges['source']).intersection(set(dailyEdges['target'])))
            users_curr = [str(int(i)) for i in users_curr]
            # print(df_dailyEdges)
            # print("Merging edges...")
            mergeEgdes = ccon.network_merge(mergeEdges, dailyEdges) # Update the mergedEdges every day
            # df_KB = pd.concat([df_KB, df_daily])  # This is the new KB Dataframe with merged posts

            G = nx.DiGraph()
            G.add_edges_from(mergeEgdes)

            conductanceList.append(Conductance(G, expertUsers, users_curr))
            prList.append(centralities(G, 'core', users_curr))
            # nbrsList.append(centralities(G, 'OutDegree', users_curr))

    # postsDailyDf['conductance'] = conductanceList
    featDF['date'] = datesList
    featDF['conductance'] = conductanceList
    featDF['core'] = prList
    # featDF['outdegree'] = nbrsList

    return featDF


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




def main():
    print(matplotlib.get_backend())
    titles = pickle.load(open('../../data/DW_data/09_15/train/features/titles_weekly.pickle', 'rb'))
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197, 220
        , 179, 219, 265, 98, 150, 121, 35, 214, 266, 89, 71, 146, 107, 64,
                           218, 135, 257, 243, 211, 236, 229, 259, 176, 159, 38]

    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulDataFiltered = vulData[vulData['forumID'].isin(forums_cve_mentions)]

    start_date = '2016-01-01'
    end_date = '2016-08-01'

    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)
    amEvents['date'] = pd.to_datetime(amEvents['date'], format="%Y-%m-%d")
    amEvents = amEvents[amEvents['date'] > pd.to_datetime('2016-08-01')]
    amEvents = amEvents[amEvents['date'] < pd.to_datetime('2016-12-01')]
    amEvents_malware = amEvents[amEvents['event_type'] == 'malicious-email']


    dates_events = list(amEvents_malware['date'])
    dates_occured = []
    for d in dates_events:
        dates_occured.append(str(d)[:10])

    # featTS_1 = pickle.load(open('../../data/DW_data/features_daysV1.0P1.pickle', 'rb'))
    # featTS_2 = pickle.load(open('../../data/DW_data/features_daysV1.0P2.pickle', 'rb'))
    #
    # featTS_2 = featTS_2.rename(columns={'conductance': 'conductance_experts', 'date': 'date_dup'})
    #
    # feat_TS = pd.concat([featTS_1, featTS_2], axis=1)
    # feat_TS = feat_TS.drop('date_dup', axis=1)
    # feat_TS = feat_TS[feat_TS['date'] >= start_date]
    # feat_TS = feat_TS[feat_TS['date'] <= end_date]

    # print(amEvents_malware)

    # forumPostsTS = countConversations(start_date, end_date, forums_cve_mentions)
    # pickle.dump(forumPostsTS, open('../../data/DW_data/posts_days_forumsV1.0.pickle', 'wb'))

    test_start_date = datetime.datetime.strptime('2016-07-24' + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

    # forumPostsTS = pickle.load(open('../../data/DW_data/posts_days_forumsV1.0.pickle', 'rb'))
    # forumPostsTS = forumPostsTS.drop_duplicates(['forum', 'date'])

    # forumPostsTS_train = feat_TS[feat_TS['date'] < test_start_date]

    feature = 'number_posts'

    forumPostsTS = pickle.load(open('../../data/DW_data/posts_days_forumsV1.0.pickle', 'rb'))
    forumPostsTS = forumPostsTS.drop_duplicates(['forum', 'date'])

    forumPostsTS_train = forumPostsTS[forumPostsTS['date'] < test_start_date]

    mat, centered_mat = formTSMatrix(forumPostsTS, feature)
    comp = getTopComponents(centered_mat, len(forums_cve_mentions))
    normSub, anomSub = projectSubspace(comp, mat)

    mat, centered_mat = formTSMatrix(forumPostsTS_train, feature)
    comp = getTopComponents(centered_mat, len(forums_cve_mentions))
    normSub, anomSub = projectSubspace(comp, mat)

    test_data = forumPostsTS[forumPostsTS['date'] >= test_start_date]
    test_data = test_data[test_data['date'] < dates_occured[len(dates_occured)-1]]
    # print(comp.shape)
    TPR, FPR = plot_ROC(normSub, anomSub, test_data, dates_occured, feature)
    # pickle.dump((TPR, FPR), open('../../data/results/ROC_feat/conductance.pickle', 'wb'))

    # TPR, FPR = pickle.load(open('../../data/results/ROC_feat/num_posts.pickle', 'rb'))
    plt.scatter(TPR, FPR)
    plt.plot(TPR, FPR)
    plt.grid()
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel('True Positive Rate', size=15)
    plt.ylabel('False Positive Rate', size=15)
    plt.subplots_adjust(left=0.17, bottom=0.17, top=0.9)

    plt.show()

    return TPR, FPR

if __name__ == "__main__":
    main()

