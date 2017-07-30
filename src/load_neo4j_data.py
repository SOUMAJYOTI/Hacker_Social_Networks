#Relationships present in the graph:
#(u:User)-[r:posted_in]->(t:Thread)e
#(t:Thread)-[r:belongs_to]->(w:Website)

# Load data from the Neo4j graph database
from neo4j.v1 import GraphDatabase, basic_auth

driver = GraphDatabase.driver("bolt://129.219.60.67:7687", auth=basic_auth("neo4j", "a$4a10c"))
session = driver.session()
uid = 94891
# thread_id = 368939
# website_id = 38

result = session.run("match (u:User)-[r:posted_in]-(t:Thread) where u.id={uid} return u,t",{"uid": uid})
for record in result:
	print("User "+str(record["u"]["id"])+" posted in thread "+str(record["t"]["id"]))

print("\n\n")

# result = session.run("match (t:Thread)-[r]-(w:Website) where t.id={tid} return t,w",{"tid":thread_id})
# for record in result:
#     print("Thread "+str(record["t"]["id"])+" belongs to website "+str(record["w"]["id"]))
#
# print("\n\n")
#
# result = session.run("match (w:Website)-[r]-(t:Thread) where w.id={wid} return t,w",{"wid":website_id})
# for record in result:
#     print("Website "+str(record["w"]["id"])+" contains thread "+str(record["t"]["id"]))