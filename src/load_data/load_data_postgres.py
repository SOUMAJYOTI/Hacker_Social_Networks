import psycopg2 as pg
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
import itertools
import matplotlib.pyplot as pplot

save_path = '../data/users_sna_cve.csv'

def get_Neo4j_data(userList):
    """ INPUT - List of user IDs"""
    """ OUTPUT - Get the user-user connection from the list of users """


if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')

    query = "select * from detailed_vuln_info_28_oct"
    print(query)
    df = pd.read_sql_query(query, con=engine)
    df.to_csv('../../data/DW_data/VulnInfo_11_17.csv')


