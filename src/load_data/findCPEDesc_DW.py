import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt


def getCPEDesc_MentionsInDW(start_date, end_date):
    results_df = pd.DataFrame()

    postsId_seen = []
    for ed in eventDesc:
        print(ed)
        results = ldap.getHackingPosts_Content(searchContent=ed, fromDate=start_date, toDate=end_date)
        count_wrong = 0
        for r_idx in range(len(results)):
            # try:
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

            item_df = pd.DataFrame(item_dict, index=[r_idx])
            results_df = results_df.append(item_df)
            # except:
            #     count_wrong += 1
            #     print(count_wrong)

    return results_df


def getImpForums(dataDf):
    df_new = dataDf.groupby(['forumsId']).count().reset_index(name="count")

    # forumsSet = df_new[]
    print(df_new)


if __name__ == "__main__":
    eventDesc = ['windows 7', 'internet explorer', 'windows vista', 'windows server', 'windows 8']
    # eventDesc = ['linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os']
    # eventDesc = ['apple', 'mac', 'mackintosh', 'mac_os', 'mac operating system']
    # eventDesc = ['google', 'google chrome', 'chrome', 'chrome OS']
    #
    # amEvenets = ['port 80',	'endpoint-malware', 'exploit_kit', 'fireeye_web_mps', 'fireeye_Hx', 'malicious-destination',
    #              'malicious-email',	'malware', 'mcafee_vse', 'pup', 'ransomware', 'trojan', 'virus', 'windows 7']
    #
    # df_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    # amEvents_df = pd.read_csv(df_path)
    # amEvents_df['date'] = pd.to_datetime(amEvents_df['date'])

    # df_date_filtered = amEvents_df[amEvents_df['date'] < pd.to_datetime('2017-10-30')]

    start_date = dt.datetime.strptime('2016-03-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-03-30', '%Y-%m-%d')

    # Get the CPE desc in Dark Web within a time period
    # results_df = getCPEDesc_MentionsInDW(start_date, end_date)
    # results_df.to_csv('../../data/Armstrong_data/Windows_IE_DW_Mar2015-sample.csv')

    results_df = pd.read_csv('../../data/Armstrong_data/Windows_IE_DW_Mar2015-sample.csv', encoding='ISO-8859-1')
    getImpForums(results_df)
