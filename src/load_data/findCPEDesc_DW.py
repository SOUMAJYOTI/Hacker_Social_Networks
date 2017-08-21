import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator

# TODO: CHECK THAT THIS RETURNS ALL THE ROWS IN DW, NOT A SAMPLE !!!!
def getCPEDesc_MentionsInDW(eventDesc, start_date, end_date):
    results_df = pd.DataFrame()

    postsId_seen = []
    count = 0
    count_data = 0
    for ed in eventDesc:
        print(ed)
        start = 0
        while True:
            print("Data count: ", count_data, " start: ", start)
            try:
                results = ldap.getHackingPosts_Content(searchContent=ed, start=start, fromDate=start_date, toDate=end_date, limNum=5000)
            except:
                break

            for r_idx in range(len(results)):
                item = results[r_idx]
                if item['postsId'] in postsId_seen:
                    continue
                postsId_seen.append(item['postsId'])
                if 'postedDate' not in item:
                    item['postedDate'] = ''
                if 'language' not in item:
                    item['language'] = ''
                item_dict = {'postsId': item['postsId'], "forumsId": item['forumsId'], 'postedDate': item['postedDate'],
                             'topicsId': item['topicId'], 'topicsName': item['topicsName'], 'language': item['language'],
                             'uid': item['uid']}

                item_df = pd.DataFrame(item_dict, index=[count])
                results_df = results_df.append(item_df)
                count += 1

            count_data += 5000
            start += 5000

    return results_df


def getImpForums(dataDf):
    dataDf['count'] = dataDf.groupby(['forumsId'])['forumsId'].transform('count')

    # forumsSet = df_new[]
    series_count = dataDf.groupby(['forumsId']).agg('count')['uid']

    count_dict = {}
    for ind, val in series_count.iteritems():
        count_dict[ind] = val

    topCount = 10
    sortedForums_count = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    for f, c in sortedForums_count:
        print(f, c)


def getArmstrongEvent_MentionsInDW(filenames, start_date, end_date, forums_list):
    results_df = pd.DataFrame()
    postsId_seen = []

    for f in filenames:
        results = ldap.getHackingPosts_Content( searchContent=f, fromDate=start_date, toDate=end_date)

        for r_idx in range(len(results)):
            # try:
            item = results[r_idx]
            print(item)
            exit()
            if item['postsId'] in postsId_seen:
                continue
            postsId_seen.append(item['postsId'])
            if 'postedDate' not in item:
                item['postedDate'] = ''
            if 'language' not in item:
                item['language'] = ''
            item_dict = {'postsId': item['postsId'], "forumsId": item['forumsId'], 'postedDate': item['postedDate'],
                         'topicsId': item['topicId'], 'topicsName': item['topicsName'],
                         'language': item['language'],
                         'uid': item['uid']}

            item_df = pd.DataFrame(item_dict, index=[r_idx])
            results_df = results_df.append(item_df)

    return results_df


if __name__ == "__main__":
    # forums_cve_mentions = [38, 113, 134, 205, 84, 159, 259, 211, 226, 150]

    # 0. Different event descriptions
    eventDesc = ['windows 7', 'internet explorer', 'windows vista', 'windows server', 'windows 8', 'operating system',
                 'linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os', 'apple', 'mac',
                 'mackintosh', 'mac_os', 'mac operating system', 'google', 'google chrome', 'chrome', 'chrome OS']
    # eventDesc = ['linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os']
    # eventDesc = ['apple', 'mac', 'mackintosh', 'mac_os', 'mac operating system']
    # eventDesc = ['google', 'google chrome', 'chrome', 'chrome OS']

    # amEvenets = ['port 80',	'endpoint-malware', 'exploit_kit', 'fireeye_web_mps', 'fireeye_Hx', 'malicious-destination',
    #              'malicious-email',	'malware', 'mcafee_vse', 'pup', 'ransomware', 'trojan', 'virus', 'windows 7']


    # 1. Get the data with CPE desc in Dark Web within a time period
    start_date = dt.datetime.strptime('2015-03-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-03-30', '%Y-%m-%d')
    results_df = getCPEDesc_MentionsInDW(eventDesc, start_date, end_date)
    results_df.to_csv('../../data/Armstrong_data/Windows_IE_DW_Jan15-Mar16.csv')

    # 2. Get the top k forums in Mar 16 that has these keywords
    # results_df = pd.read_csv('../../data/Armstrong_data/Windows_IE_DW_Mar2016-sample.csv', encoding='ISO-8859-1')
    # getImpForums(results_df)

    # 3. Find the data with Armstrong event description in Darkweb data
    # df_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    # amEvents_df = pd.read_csv(df_path)
    # amEvents_df['date'] = pd.to_datetime(amEvents_df['date'])
    # amEvents_df = amEvents_df.fillna(value='')
    #
    # results_final = pd.DataFrame()
    # # filenames_seen = []
    # start_date = dt.datetime.strptime('2016-04-01', '%Y-%m-%d')
    # end_date = dt.datetime.strptime('2016-05-01', '%Y-%m-%d')
    # amEvents_df_slice = amEvents_df[amEvents_df['date'] < end_date]

    # fNames_selected = ['trojan']
    # fNames_selected = ['trojan', 'mcafee_vse', 'fireeye_web_mps', 'ransomware']
    # results_final = getArmstrongEvent_MentionsInDW(fNames_selected, start_date, end_date, [])
    # results_final.to_csv('../../data/Armstrong_data/Armstrong_selectedNames_Apr-May2016-sample.csv')

    # for filename in fNames_selected:

    # for i, r in amEvents_df_slice.iterrows():
    #     date_event = r['date']
    #     filename = r['detector']

    #     if filename in filenames_seen or filename == '' :
    #         continue
    #
    #     filenames_seen.append(filename)
    #     print(filename)
    #
    #     r = getArmstrongEvent_MentionsInDW(fNames_selected, start_date, end_date, [])
    #     results_final.append(r)
    #     print(r)
    #     print('hello', results_final)
    #
    # results_final.to_csv('../../data/Armstrong_data/Armstrong_selectedNames_Apr-May2016-sample.csv')

