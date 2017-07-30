import collections

import time

from jupyter_client._version import protocol_version
from numpy.distutils.system_info import numarray_info
import requests
#from DB_connection import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from multiprocessing import Pool
import pickle

def call_API(url):
    print (url)
    headers = {"userId" : "labuser", "apiKey" : "fc40c488-9cb8-4e50-9c0c-14af4e6098da"}
    response = requests.get(url, headers=headers)
    try:
        #df = pd.DataFrame.from_dict(response.json())
        return response.json()['results']
    except:
        raise ValueError(response.json()['message'])


def graphToDataframe(graph):
    data={}
    data['cve']= [x for x in graph]
    other_cols = graph.node.get(graph.node.keys()[0]).keys()
    for key in other_cols:
        data[key] = [graph.node.get(x)[key] for x in graph]
    return pd.DataFrame(data)


def setStands():
    font = {'size': 28}
    sns.set_context("poster", font_scale=1.4)
    plt.rc('font', **font)


def getUid(usersid):
    headers = {"userId": "labuser", "apiKey": "fc40c488-9cb8-4e50-9c0c-14af4e6098da"}
    response = requests.get("https://apiGargoyle.com/GargoyleApi/getUid?usersId=" + str(usersid), headers=headers)
    try:
        # df = pd.DataFrame.from_dict(response.json())
        return response.json()['uid']
    except:
        raise ValueError(response.json()['message'])
    pass


def dw_sna():



    #dwCwes=getFromDB("select cve, darkweb_date, published_date, usersid::text, vendorid::text, product, cwe from complete where has_dw = true and published_date > '2014-12-31'")

    dwVulnUsers = getFromDB('SELECT usersid::text, forumsid::text 	FROM public."postsVulns"')


    vulnUsers =list(set(dwVulnUsers.usersid.tolist()))
    listOfUsers =[]
    for user in vulnUsers[:]:
        print ('user :%s \n', str(user), vulnUsers.index(user))
        try:
            if user != None:
                uid = getUid(str(user))
                listOfConnections = call_API("https://apigargoyle.com/GargoyleApi/generateUserNet?limit=10000&uid="+ str(uid))
                listOfSites= call_API("https://apigargoyle.com/GargoyleApi/generateForumMarketNet?limit=10000&uid=" + str(uid))
                userPosts = call_API("https://apigargoyle.com/GargoyleApi/getUserPosts?limit=10000&uid=" + str(uid))
                #forumsData = call_API("https://apigargoyle.com/GargoyleApi/getForumStatistics?limit=10000&forumsId=" + str(dwVulnUsers.loc[dwVulnUsers['usersid']==user]['forumsid'].values[0]))

                numOfConnections = len(listOfConnections)
                count = 0

                for i in listOfConnections:
                    #userb= i[1][7:]
                    #print 'test'
                    count+= int(i[2][7:])

                listOfUsers.append((int(user), uid, len(listOfSites),numOfConnections, count, len(userPosts)))
        except:
            print ('this user was not processed\n' + sys.exc_info()[0])

    return listOfUsers

    #print 'test'


def main():

    pass

def remove_adjacent(nums):
    return [a for a, b in zip(nums, nums[1:] + [not nums[-1]]) if a != b]


def computeEdges(posts, forumid, topicid):
    uids = remove_adjacent(posts['uid'].tolist())
    list = []
    for i in range(len(uids)):
        source = uids[i]
        for j in range(len(uids[i + 1:i + 11])):
            target = uids[i + 1 + j]
            if source != target:
                if not (source, target, posts.iloc[i]['posteddate'], forumid, topicid) in list:
                    list.append((source, target, posts.iloc[i]['posteddate'], forumid, topicid))

    return list

def processForum(df):

    edges = []
    t1 = time.time()
    forumid = df['forumsid'].iloc[0]
    topics = df['topicid'].unique()
    print ('forum: ' + str(forumid) + ', topics: '+ str(len(topics)) + ', posts: ')
    for t in topics:
        posts = df[df['topicid'] == t].sort_values(['posteddate', 'postedtime'], ascending=[0, 0])
        edges.extend(computeEdges(posts, forumid, t))

    print ('Done with forum ' + str(forumid) + '. It took: ' + (str(time.time() - t1) + ' to processes topics: '+ str(len(topics))))
    return edges

def processTopic(df):
    edges = []
    t1 = time.time()
    forumid = df['forumsid'].iloc[0]
    topics = df['topicid'].unique()[0]
    #print ('forum: ' + str(forumid) + ', topics: '+ str(len(topics)) + ', posts: ')

    posts = df[df['topicid'] == topics].sort_values(['posteddate', 'postedtime'], ascending=[0, 0])
    edges.extend(computeEdges(posts, forumid, topics))

    print ('Done with forum ' + str(forumid) + '. Topic:' + str(topics) + 'It took: ' + (str(time.time() - t1) ))
    return edges

def computeConnections(df):
    t1 = time.time()
    edges =[]
    l = []
    '''
    topics = df['topicid'].unique()



    for t in topics:
        l.append(df[df['topicid'] == t])
    '''


    forums = df['forumsid'].unique()
    print (forums)
    for f in forums:
        l.append(df[df['forumsid'] == f])


    pool = Pool()
    results = pool.map(processForum, l)
    pool.close()
    pool.join()
    t2 = time.time() - t1
    print ('time taken to finsh: ' + str(t2))
    '''
    for f in forums:
        print 'Now, we have ' + str(len(edges)) + ' edges'
        edges.extend(processForum(f))
    '''
    for i in results:
        edges.extend(i)
    df3 = pd.DataFrame(edges, columns= ['source', 'target', 'date', 'forumid', 'topicid'])
    df3.to_pickle('DW user edges DataFrame multiprocessed 2010 - june2016 only 10 users.pickle')
    change_pickle_protocol('DW user edges DataFrame multiprocessed 2010 - june2016 only 10 users.pickle')

    pass



def change_pickle_protocol(filepath,protocol=2):
    with open(filepath,'rb') as f:
        obj = pickle.load(f)
    with open(filepath,'wb') as f:
        pickle.dump(obj,f,protocol=protocol)


if __name__ == "__main__":

    # set the last date of traning
    lastDatyeInTraning = "2016-05-30"
    #set the first date to construct social graph
    startDate = "2010-01-01"
    setStands()

    from sqlalchemy import create_engine

    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cve')

    query= "select forumsid, topicid, posteddate::date, postedtime::time, postsid, uid from dw_posts where posteddate::date > '"\
           + startDate+ "' and posteddate::date < '" + lastDatyeInTraning + "' and forumsid in (SELECT" \
                                                                            "  distinct forumsid FROM detailed_vuln_info) " \
                                                                            "and topicid not in (select topicid from dw_posts group by 1 having count(*) > 200)"
    print (query)
    df = pd.read_sql_query(query,con=engine)
    #df = getFromDB(query)

    computeConnections(df)
    print ('test')

    #listOfUsers = main()
    df = pd.DataFrame(listOfUsers , columns=['usersid', 'uid', 'num_sites', 'num_conn', 'sum_conn_weights', 'num_posts'])


    df.to_sql('dw_users_sna', engine)