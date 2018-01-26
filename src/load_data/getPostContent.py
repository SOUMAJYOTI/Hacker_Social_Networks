import pycyr3con
import json
import requests
from sqlalchemy import create_engine
import pandas as pd
import time

engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')


def initiate_headers():
    global headers
    print('------' \
          'updating headers' \
          '------')
    response = requests.post("https://apigargoyle.com/GargoyleApi/login?userId=mbalasu9&password=6F4E50vi9Zh4")
    headers = {"userId": "mbalasu9", "apiKey": json.loads(response.text)['apiKey']}
    pass


def test_API(url='https://apigargoyle.com/GargoyleApi/getVulnerabilityInfo?vulnerabilityId=cve-2012-1864'):
    print(url)
    global headers
    if headers == None:
        initiate_headers()
    try:
        response = requests.get(url, headers=headers)
        return response.json()['results']
    except:
        print(response.json())
        if response.json()['error'] == "Unauthorized":
            initiate_headers()
            response = requests.get(url, headers=headers)
    return response.json()['results']



def getHackingPosts(searchTerm):
    results = test_API("https://apigargoyle.com/GargoyleApi/getHackingPosts?limit=2000&postContent=" + searchTerm)
    df = pd.DataFrame.from_dict(results)
    flag = True
    start = len(df)
    while (flag):
        time.sleep(1)
        try:
            flag = False
            results = test_API(
                "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit=5000&start=" + str(start))

            if len(results) == 0:
                flag = False
            else:
                start += len(results)
            df = df.append(pd.DataFrame.from_dict(results))
        except:
            print('didnt work...')
            time.sleep(5)
    df2 = df.copy()
    for c in df.columns:
        try:
            df2[c] = df[c].astype(str)
        except:
            print(c)

    #change the DB name if you need
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    df2.columns = [c.lower() for c in df.columns]  # postgres doesn't like capitals or spaces
    df2['dates'] = pd.to_datetime("'2017-11-06'".replace("'","")).dt.date
    #df2.to_sql('hacking_posts', engine, if_exists='fail')

    return df2


if __name__ == "__main__":
    frDate = "2016-01-01"
    toDate = "2016-01-05"

    result = getHackingPosts_Date(frDate, toDate)

    print(result)
