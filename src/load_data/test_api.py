import datetime as dt
import pandas as pd
import requests

def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum


def getHackingPosts(searchContent='', fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+"&from="+dateToString(fromDate)\
          +"&to="+dateToString(toDate)+ "&postContent="+str(searchContent)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


if __name__ == "__main__":
    print('Starting to fetch data from server...')
    #
    # itemLim = 5000
    # clusterList = {}
    #
    # # Hacking Posts Statistics
    # fileName = 'HackingPostsStatistics'
    start_date = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-01-20', '%Y-%m-%d')
    # hp = getHackingPosts(searchContent='Windows', fromDate=start_date, toDate=end_date, limNum=itemLim)
    #
    # hp_df = pd.DataFrame()
    # hpList = {}
    # count_wrong = 0
    # for hp_idx in range(len(hp)):
    #     try:
    #         item_df = pd.DataFrame(hp[hp_idx], index=[hp_idx])
    #         print(hp[hp_idx])
    #         # clusterList = hp[hp_idx]
    #         # item_df = pd.DataFrame(clusterList, )
    #         # hp_df = hp_df.append(item_df)
    #
    #     except:
    #         print(hp[hp_idx])
    #         exit()
    #         count_wrong += 1
    # print(count_wrong)

    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit=" + str(5000) + "&from=" + dateToString(start_date) \
          + "&to=" + dateToString(end_date) + "&postContent=" + str('windows')
    headers = {"userId": "labuser", "apiKey": "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers).json()['results']

    for idx in range(len(response)):
        print(response[idx])