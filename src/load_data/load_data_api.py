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


def getVulnerabilityInfo(start=0, fromDate=dt.date.today(), toDate=dt.date.today(), limNum=5000):
    url = 'https://apigargoyle.com/GargoyleApi/getDetailedVulnerabilityInfo?limit=' + str(limNum) + "&start=" + str(start)
    headers = {"userId": "labuser", "apiKey": "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getUsersForums(userId, limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+"&usersId="+userId
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getHackingPosts(start=0, fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0, fId=40):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit=" + str(limNum) + \
          "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)+"&forumsId=" + str(fId) + "&start=" + str(start)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


def getHackingPosts_Content(searchContent='', start=0, fromDate=dt.date.today(),  toDate=dt.date.today(),limNum=5000):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+\
          "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)+ "&postContent="+str(searchContent) + "&start="+str(start)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    if 'results' not in response.json():
        return {}
    else:
        return response.json()['results']

def getHackingPosts_Content_Forums(fId, searchContent='', fromDate=dt.date.today(),  toDate=dt.date.today(),
                                   start=0, limNum=5000):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+\
          "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)+ "&postContent="+str(searchContent)\
          + "&forumsId="+ str(fId) + "&start=" + str(start)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    if 'results' not in response.json():
        return {}
    else:
        return response.json()['results']


def getHackingThreads_Content(searchContent='', fromDate=dt.date.today(),  toDate=dt.date.today(),limNum=5000):
    url = "https://apigargoyle.com/GargoyleApi/getHackingThreads?limit="+str(limNum)+\
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


