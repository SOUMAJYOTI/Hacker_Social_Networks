import load_data_api as ldap
import pandas as pd
from sqlalchemy import create_engine
import datetime as dt


def getDW_Data(forums_list, start_date, end_date):
    results_df = pd.DataFrame()
    postsId_seen = []

    count_data = 0
    for f in forums_list:
        print("Forum: ", f)
        start = 0
        while True:
            print("Data count: ", count_data, " start: ", start)
            try:
                results = ldap.getHackingPosts(start=start, limNum=5000, fId=f, fromDate=start_date, toDate=end_date)
            except:
                break

            if len(results) == 0:
                break

            # if count_data > 10000:
            #     break

            for r_idx in range(len(results)):
                # try:
                item = results[r_idx]

                # if item['postsId'] in postsId_seen:
                #     continue
                # postsId_seen.append(item['postsId'])
                if 'postedDate' not in item:
                    item['postedDate'] = ''
                if 'language' not in item:
                    item['language'] = ''
                if 'postContent' not in item:
                    item['postContent'] = ''
                item_dict = {'postsId': item['postsId'], "forumsId": item['forumsId'], 'postedDate': item['postedDate'],
                             'topicsId': item['topicId'], 'topicsName': item['topicsName'],
                             'language': item['language'], 'postContent': item['postContent'],
                             'uid': item['uid']}

                item_df = pd.DataFrame(item_dict, index=[count_data + r_idx])
                results_df = results_df.append(item_df)

            count_data += 5000
            start += 5000

    return results_df


if __name__ == "__main__":
    forums_cve_mentions = [38, 113, 134, 205, 84, 159, 259, 211, 226, 150]
    # forums_cve_mentions = [38, 113, 134]

    start_date = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-12-31', '%Y-%m-%d')

    results_final = getDW_Data(forums_cve_mentions, start_date, end_date)
    # results_final.to_csv('DW_data_f_38_113_134_205_84_159_259_211_226_150.csv')

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/DarkWeb_Soum')
    results_final.to_sql('DW_forums_topCVE_mention', engine, if_exists='append')
