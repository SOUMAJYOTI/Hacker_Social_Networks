# Load data from the Neo4j graph database
from neo4j.v1 import GraphDatabase, basic_auth
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    driver = GraphDatabase.driver("bolt://129.219.60.67:7687", auth=basic_auth("neo4j", "a$4a10c"))
    session = driver.session()

    # Get all the users in Neo4j database
    userList = []
    result = session.run("match (u:User) return u")
    for record in result:
        userList.append(record["u"]["id"])

    # For each user query the threads that user posted in
    user_thread_dict = {}
    query_User_threads = "match (u:User)-[r]-(p:Post)-[r2]-(t:Thread) where u.id={uid} return u, p, t"
    user_threadCount = {}
    user_count = 0
    for uid in userList:
        print("Querying for User: ", user_count)
        result = session.run(query_User_threads, {"uid": uid})
        for record in result:
            if uid not in user_thread_dict:
                user_thread_dict[uid] = 1
            else:
                user_thread_dict[uid] += 1
            # print("User " + str(record["u"]["id"]) + " posted in post " + str(record["p"]["id"]) + " of thread " + str(record["t"]["id"]))
        user_count += 1
        # if user_count > 50:
        #     break
    df_thread_count = pd.DataFrame(list(user_thread_dict.values()), columns=['ThreadCounts'])

    print("Total number of users: ", user_count)
    # print(df_thread_count)

    df_thread_count.plot.hist(by='ThreadCounts', bins=15)
    plt.grid(True)
    plt.xlabel('# threads by each user', fontsize=30)
    plt.ylabel('# Users (log) ', fontsize=30)
    plt.yscale('log')
    plt.tick_params('x', labelsize=25)
    plt.tick_params('y', labelsize=25)
    plt.show()

