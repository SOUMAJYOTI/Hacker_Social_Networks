import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pickle
import datetime as dt


if __name__ == "__main__":
    dataLim = 10000
    countData = 0
    startDate = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    endDate = dt.datetime.strptime('2016-04-01', '%Y-%m-%d')
    vIds = []
    pDates = []
    marketPlaces = []
    indic = []
    itemNames = []
    forums = []

    while countData < dataLim:
        results = ldap.getVulnerabilityInfo(start=countData, fromDate=dt.date.today(), toDate=dt.date.today(), limNum=5000)

        for r_idx in range(len(results)):
            item = results[r_idx]

            vIds.append(item["vulnerabilityId"])
            pDates.append(item["postedDate"])
            if item["indicator"] == "Item":
                indic.append("Item")
                marketPlaces.append(item['marketPlaceId'])
                itemNames.append(item['itemName'])
                forums.append("NA")
            else:
                indic.append("Post")
                marketPlaces.append("NA")
                itemNames.append("NA")
                forums.append(item["forumsId"])


