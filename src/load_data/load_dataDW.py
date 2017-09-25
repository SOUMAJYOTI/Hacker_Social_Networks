import load_data_api as ldap
import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
import pickle


def getDW_Data(forums_list, start_date, end_date):
    results_df = pd.DataFrame()
    postsId_seen = []

    count_data = 0
    for f in forums_list:
        print("Forum: ", f)
        start = 0
        while True:
            print("Data count: ", count_data, " start: ", start)
            # try:
            results = ldap.getHackingPosts(start=start, limNum=5000, fId=f, fromDate=start_date, toDate=end_date)
            # except:
            #     break

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


def getDW_data_postgres(forums_list, start_date, end_date):
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')

    results = pd.DataFrame()

    cumulative = []
    for f in forums_list:
        query = "select forumsid, topicid, posteddate::date, postedtime::time, postsid, uid from dw_posts where posteddate::date > '" \
        + start_date + "' and posteddate::date < '" + end_date + "' and forumsid = " + str(f)

        df = pd.read_sql_query(query, con=engine)
        cumulative.append(df)

    results_final = pd.concat(cumulative)
    return results_final

if __name__ == "__main__":
    forums_cve_mentions = [88, 248, 133, 49, 62, 161, 84, 60, 104, 173, 250, 105, 147, 40, 197]

    # start_date = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    # end_date = dt.datetime.strptime('2016-03-01', '%Y-%m-%d')

    start_date = '2016-03-01'
    end_date = '2016-04-01'

    results_final = getDW_data_postgres(forums_cve_mentions, start_date, end_date)
    pickle.dump(results_final, open('../../data/DW_data/09_15/DW_data_selected_forums_Apr16.pickle', 'wb'))

    # results_final = getDW_Data(forums_cve_mentions, start_date, end_date)
    # results_final.to_csv('../../data/DW_data/08_29/DW_data_selected_forums_Jan-Mar16.csv')
    # pickle.dump(results_final, open('../../data/DW_data/08_29/DW_data_selected_forums_Jan-Mar16.csv', 'wb'))

    # engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/DarkWeb_Soum')
    # results_final.to_sql('DW_forums_topCVE_mention', engine, if_exists='append')
