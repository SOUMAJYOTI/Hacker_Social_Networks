import requests


def getVulnerabilityInfo(start=0, limNum=5000):
    url = 'https://apigargoyle.com/GargoyleApi/getDetailedVulnerabilityInfo?limit=' + str(limNum) + "&start=" + str(start)
    headers = {"userId": "labuser", "apiKey": "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


if __name__ == "__main__":
    resultsCount = 0
    countData = 0

    while True:
        print("Start: ", countData)
        results = getVulnerabilityInfo(start=countData, limNum=2000)

        if len(results) == 0:
            break

        resultsCount += len(results)
        countData += 2000

    print("Number of results returned..", resultsCount)

