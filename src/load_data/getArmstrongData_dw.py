import datetime as dt
import pandas as pd
import requests
import load_armstrong_data as lam
import load_data_sdk as lds
import load_data_api as lda


def getHackingPosts(limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

if __name__ == "__main__":
    fPaths = ['../../data/Armstrong_data/release/release/data.json',
              '../../data/Armstrong_data/release/release2/data.json']

    df_am = lam.load_am_data(fPaths)
    # list_attack_files = df_am["attack_file"].tolist()

    write_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    df_am.to_csv(write_path)
    exit()

    # start_date = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    # end_date = dt.datetime.strptime('2017-01-20', '%Y-%m-%d')
    #
    # for lf in list_attack_files:
    #     print("\nFilename: ", lf)
    #     result = lda.getHackingPosts_Content(lf)
    #     print(result)
    #
    #
