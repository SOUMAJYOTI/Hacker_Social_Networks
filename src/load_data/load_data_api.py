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

def getUsersForums(userId, limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+"&usersId="+userId
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getHackingPosts(fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0, fId=40):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+\
          "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)+"&forumsId="+str(fId)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


def getHackingPosts_Content(searchContent='', fromDate=dt.date.today(),  toDate=dt.date.today(),limNum=5000):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+\
          "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)+ "&postContent="+str(searchContent)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    if 'results' not in response.json():
        return {}
    else:
        return response.json()['results']


def getHackingItems(fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingItems?limit="+str(limNum)+"&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getClusterStatistics(clusterName='', limNum=0):
    if clusterName == '':
        url = "https://apigargoyle.com/GargoyleApi/getClusterStatistics?limit=" + str(
            limNum)
    else:
        url = "https://apigargoyle.com/GargoyleApi/getClusterStatistics?limit=" + str(limNum) + "&clusterName=" + clusterName
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


# if __name__ == "__main__":
#     print('Starting to fetch data from server...')
#     day = dt.date.today()
#     filePath = r"../darkweb_data/"
#
#     itemLim = 1000
#     # Cluster Statistics
#     fileName = 'clusterStatistics'
#     # cls = getClusterStatistics()
#     # cls_df = pd.DataFrame()
#     # clusterList = {}
#     # for cl_idx in range(len(cls)):
#     #     clusterList = cls[cl_idx]
#     #     item_df = pd.DataFrame(clusterList, index=[cl_idx])
#     #     cls_df = cls_df.append(item_df)
#     # cls_df.to_csv(filePath+fileName+'.csv')
#
#     # Hacking Posts Statistics
#     fileName = 'HackingPostsStatistics'
#     start_date = dt.datetime.strptime('2014-10-15', '%Y-%m-%d')
#     end_date = dt.datetime.strptime('2016-10-15', '%Y-%m-%d')
#     hp = getHackingPosts(fromDate=start_date, toDate=end_date, limNum=itemLim)
#     hp_df = pd.DataFrame()
#     hpList = {}
#     for hp_idx in range(len(hp)):
#         try:
#             clusterList = hp[hp_idx]
#             item_df = pd.DataFrame(clusterList)
#             hp_df = hp_df.append(item_df)
#         except:
#             # print(hp[hp_idx])
#             continue
#
#     # hp_df.to_csv(fileName+'.csv')
#
#     # # Major Forum Statistics
#     # fileName = 'HackingPostsStatistics_MajorForum'
#     # forums_count_df = hp_df.groupby('forumsId').apply(lambda x: x.drop('forumsId', axis=1).drop_duplicates().shape[0]).reset_index()
#     # max_forumsId = forums_count_df[0].argmax()
#     # hp_max_forums = hp_df[hp_df['forumsId'] == forums_count_df['forumsId'][max_forumsId]]
#     # hp_max_forums.to_csv(filePath+fileName+'.csv')
#     #
#     # # Hacking items statistics
#     # fileName = 'HackingItemsStatistics'
#     # start_date = dt.datetime.strptime('2016-08-15', '%Y-%m-%d')
#     # end_date = dt.date.today()
#     # hi = getHackingItems(fromDate=start_date, toDate=end_date, limNum=itemLim)
#     # hi_df = pd.DataFrame()
#     # hiList = {}
#     # for hi_idx in range(len(hi)):
#     #     clusterList = hi[hi_idx]
#     #     item_df = pd.DataFrame(clusterList, index=[hi_idx])
#     #     hi_df = hi_df.append(item_df)
#     # hi_df.to_csv(filePath+fileName+'.csv')
#     #
#     # # Major Marketplace statistics
#     # fileName = 'HackingItemstatistics_MajorMarketplace'
#     # mp_count_df = hi_df.groupby('marketplaceId').apply(lambda x: x.drop('marketplaceId', axis=1).drop_duplicates().shape[0]).reset_index()
#     # max_mpId = mp_count_df[0].argmax()
#     # hi_max_forums = hi_df[hi_df['marketplaceId'] == mp_count_df['marketplaceId'][max_mpId]]
#     # hi_max_forums.to_csv(filePath+fileName+'.csv')
#
#
#     print('Done loading data...')
#
#     # find the largest community for the forums

