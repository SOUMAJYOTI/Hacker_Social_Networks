import pycyr3con
import json


def getUserPosts(uId, limit=1000):
    """ uid - user ID """
    api = pycyr3con.Api(userId='labuser', apiKey='a9a2370f-4959-4511-b263-5477d31329cf')
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 1000:
            result.append(api.getUserPosts(usersId=uId, limit=limit))
        else:
            result.extend(api.getUserPosts(usersId=uId, start=start_ptr, limit=1000))
            start_ptr += 10000

    return result


def getHackingPosts_Date(frDate, toDate, limit=1000):
    api = pycyr3con.Api(userId='Soumajyoti', apiKey='FuXB4lL74N(')
    start_ptr = 0
    result = []
    while True:
        try:
            result.extend(api.getHackingPosts(fromDate=frDate, toDate=toDate, start=start_ptr, limit=start_ptr+limit))
            start_ptr += 1000
        except:
            break

    return result


def getHackingPosts(limit=3000):
    api = pycyr3con.Api(userId='Soumajyoti', apiKey='FuXB4lL74N(')
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 3000:
            result.append(api.getHackingPosts(limit=limit))
        else:
            result.append(api.getHackingPosts(start=start_ptr, limit=start_ptr+3000))
            start_ptr += 3000

    return result


def detailedVulnInfo(start_date, end_date, limit=1000):
    api = pycyr3con.Api(userId='Soumajyoti', apiKey='FuXB4lL74N(')
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 1000:
            result.append(api.getDetailedVulnerabilityInfo(limit=limit))
        else:
            result.append(api.getDetailedVulnerabilityInfo(limit=limit))
            start_ptr += 1000

    return result

if __name__ == "__main__":
    frDate = "2016-01-01"
    toDate = "2016-01-05"

    result = getHackingPosts_Date(frDate, toDate)

    print(result)

