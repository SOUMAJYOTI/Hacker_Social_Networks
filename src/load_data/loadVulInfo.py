import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pickle
import datetime as dt
import pandas as pd

if __name__ == "__main__":
    dataLim = 30000
    countData = 0
    # startDate = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')
    # endDate = dt.datetime.strptime('2017-01-01', '%Y-%m-%d')
    vIds = []
    pDates = []
    marketPlaces = []
    indic = []
    itemNames = []
    forums = []
    softwareTags = []
    financialTags = []
    numUsers = []
    users = []

    while True:
        print("Count_Data: " , countData)
        # try:
        results = ldap.getVulnerabilityInfo(start=countData, limNum=5000)
        print(len(results))
        # except:
        #     break

        if len(results) == 0:
            break
        for r_idx in range(len(results)):
            item = results[r_idx]

            if "postedDate" not in item:
                pDates.append('')
            elif item["postedDate"] == "None":
                pDates.append('')
            else:
                pDates.append(item["postedDate"])
            vIds.append(item["vulnerabilityId"])

            if "softwareTags" in item:
                softwareTags.append(item['softwareTags'])
            else:
                softwareTags.append('NA')

            if "financialTags" in item:
                financialTags.append(item['financialTags'])
            else:
                financialTags.append('NA')

            if "noOfUsers" in item:
                numUsers.append(item["noOfUsers"])
            else:
                numUsers.append('NA')

            if "uids" in item:
                users.append(item['uids'])

            if item["indicator"] == "Item":
                indic.append("Item")
                marketPlaces.append(item['marketPlaceId'])
                if "itemDescription" in item:
                    itemNames.append(item['itemDescription'])
                else:
                    itemNames.append('')
                forums.append("NA")
            else:
                indic.append("Post")
                marketPlaces.append("NA")
                itemNames.append(item["postContent"])
                forums.append(item["forumsId"])

        countData += 5000

    df = pd.DataFrame()
    df["postedDate"] = pDates
    df["postedDate"] = df["postedDate"].astype('datetime64')
    df["vulnId"] = vIds
    df["indicator"] = indic
    df["marketId"] = marketPlaces
    df["forumID"] = forums
    df['itemName'] = itemNames
    df['softwareTags'] = softwareTags
    df['financialTags'] = financialTags
    df['numUsers'] = numUsers
    df['users'] = users

    pickle.dump(df, open("../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle", 'wb'))

    print(df)