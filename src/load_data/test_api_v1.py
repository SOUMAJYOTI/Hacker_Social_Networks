import requests
import datetime as dt

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
    url = 'https://apigargoyle.com/GargoyleApi/getDetailedVulnerabilityInfo?limit=' + str(limNum) + "&start=" + str(start)\
          + "&from=" + dateToString(fromDate) + "&to=" + dateToString(toDate)
    headers = {"userId": "labuser", "apiKey": "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


def getHackingPosts(start=0, fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0, fId=40):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+\
          "&forumsId="+str(fId) + "&start="+str(start)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

if __name__ == "__main__":
    resultsCount = 0
    countData = 0

    start_date = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-03-01', '%Y-%m-%d')

    while True:
        print("Start: ", countData)
        results = getVulnerabilityInfo(start=countData, fromDate=start_date, toDate=end_date, limNum=5000)

        # for r_idx in range(len(results)):
        #     # try:
        #     item = results[r_idx]
        #     print(item['postedDate'])
        #
        # exit()
        if len(results) == 0:
            break

        resultsCount += len(results)
        countData += 5000

    print("Number of results returned..", resultsCount)

