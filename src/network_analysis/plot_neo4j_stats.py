# Load data from the Neo4j graph database
from neo4j.v1 import GraphDatabase, basic_auth
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pickle


def users_neighbors_simple(session):
    threadList = []
    result = session.run("match (t:Thread) return t")
    for record in result:
        threadList.append(record["t"]["id"])

    threadList = list(set(threadList))

    # For each user query the numbers of neighbors
    query_user_nbrs = "MATCH (s:User)-[r1]-(p1:Post)-[r2]-(t:Thread) where t.id={tid} return s"

    threadcount = 0
    user_edges = []
    for tid in threadList:
        print("Thread: ", threadcount)
        user_list = []
        result = session.run(query_user_nbrs, {"tid": tid})
        for record in result:
            uid = record["s"]["id"]
            user_list.append(uid)

        print(len(user_list))
        # pair_users = list(itertools.combinations(user_list, 2))

        for i in range(len(user_list)):
            for j in range(i+1, len(user_list)):
                user_edges.append((i, j))
        # for u, v in pair_users:
        #     if (u, v) in user_edges or (v, u) in user_edges:
        #         continue
        #     user_edges.append((u, v))

        threadcount += 1
        # if threadcount > 10:
        #     break

    pickle.dump(user_edges, open('user_edges.pickle', 'wb'))


def users_neighbors_neo4j(session):
    # Get all the users in Neo4j database
    userList = []
    result = session.run("match (u:User) return u")
    for record in result:
        userList.append(record["u"]["id"])

    # For each user query the numbers of neighbors
    user_thread_dict = {}
    query_user_nbrs = "MATCH (s:User)-[r1:posted_in]->(p1:Post)-[r2:belongs_to]->(t:Thread)<-[r3:belongs_to]" \
                      "-(p2:Post)<-[r4:posted_in]-(tr:User) " \
                      "where s.id={uid} return tr"

    forumCount = {}
    user_count = 0
    for uid in userList:
        print("Querying for User: ", user_count)
        result = session.run(query_user_nbrs, {"uid": uid})
        num_nbrs = list(result)
        print(num_nbrs)
        # if num_forums not in forumCount:
        #     forumCount[num_forums] = 1
        # else:
        #     forumCount[num_forums] += 1

            # for record in result:
            #     if uid not in user_thread_dict:
            #         user_thread_dict[uid] = 1
            #     else:
            #         user_thread_dict[uid] += 1
            # print("User " + str(record["u"]["id"]) + " posted in post " + str(record["p"]["id"]) + " of thread " + str(record["t"]["id"]))
        user_count += 1
        if user_count > 50:
            break


def user_forum_stats(session):
    # Get all the users in Neo4j database
    userList = []
    result = session.run("match (u:User) return u")
    for record in result:
        userList.append(record["u"]["id"])

    userList = list(set(userList))
    print(len(userList))
    # For each user query the forums that user posted in
    user_thread_dict = {}
    # query_User_threads = "match (u:User)-[r]-(p:Post)-[r2]-(t:Thread)-[r3]-(f:Forum) " \
    #                      "where u.id={uid} return u, p, t, f"
    query_User_forums = "match (u:User)-[r]-(p:Post)-[r2]-(t:Thread)-[r3]-(f:Forum) " \
                         "where u.id ={uid} return COUNT(distinct f)"
    forumCount = {}
    user_count = 0
    for uid in userList:
        print("Querying for User: ", user_count)
        result = session.run(query_User_forums, {"uid": uid})
        num_forums = list(result)[0][0]
        if num_forums not in forumCount:
            forumCount[num_forums] = 1
        else:
            forumCount[num_forums] += 1
        user_count += 1
        # if user_count > 50:
        #     break

    print(forumCount)


if __name__ == "__main__":
    driver = GraphDatabase.driver("bolt://129.219.60.67:7687", auth=basic_auth("neo4j", "a$4a10c"))
    session = driver.session()

    # user_forum_stats(session)
    users_neighbors_simple(session)


    # print(df_thread_count)

    # df_thread_count.plot.hist(by='ThreadCounts', bins=30)
    # plt.grid(True)
    # plt.xlabel('# threads by each user', fontsize=30)
    # plt.ylabel('# Users', fontsize=30)
    # plt.tick_params('x', labelsize=25)
    # plt.tick_params('y', labelsize=25)
    # plt.show()

