import datetime as dt
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


if __name__ == "__main__":
    print('Starting to fetch data from server...')

    start_date = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-01-30', '%Y-%m-%d')

    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit=" + str(500) + "&from=" + dateToString(start_date) \
          + "&to=" + dateToString(end_date) +"&forumsId=" + str(88) + "&start=" + str(0)
    headers = {"userId": "labuser", "apiKey": "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers).json()['results']

    for idx in range(len(response)):
        print(response[idx]['forumsId'], response[idx]['postedDate'])