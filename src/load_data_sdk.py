import pycyr3con
import json

def get_UserPosts(uId, limit=10000):
    """ uid - user ID """
    api = pycyr3con.Api()
    start_ptr = 0
    result = []
    while start_ptr < limit:
        if limit < 10000:
            result.append(api.getUserPosts(usersId=uId, limit=limit))
        else:
            result.extend(api.getUserPosts(usersId=uId, start=start_ptr, limit=10000))
            start_ptr += 10000

    return result

