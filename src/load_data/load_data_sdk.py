import pycyr3con
import json

def getUserPosts(uId, limit=10000):
    """ uid - user ID """
    api = pycyr3con.Api(userId='labuser', apiKey='a9a2370f-4959-4511-b263-5477d31329cf')
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 10000:
            result.append(api.getUserPosts(usersId=uId, limit=limit))
        else:
            result.extend(api.getUserPosts(usersId=uId, start=start_ptr, limit=10000))
            start_ptr += 10000

    return result


def getHackingPosts_Date(frDate, toDate, content='', limit=10000):
    api = pycyr3con.Api(userId='labuser', apiKey='a9a2370f-4959-4511-b263-5477d31329cf')
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 10000:
            result.append(api.getHackingPosts(fromDate=frDate, toDate=toDate, postContent=content, limit=limit))
        else:
            result.append(api.getHackingPosts(fromDate=frDate, toDate=toDate, limit=limit))
            start_ptr += 10000

    return result


def getHackingPosts_Content(searchContent, limit=10000):
    api = pycyr3con.Api(userId='labuser', apiKey='a9a2370f-4959-4511-b263-5477d31329cf')
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 10000:
            result.append(api.getHackingPosts(limit=limit))
        else:
            result.append(api.getHackingPosts(limit=limit))
            start_ptr += 10000

    return result