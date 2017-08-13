import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt

if __name__ == "__main__":
    subevents = ['FlashPlayer2015+', 'PHP_2015+', 'Win7+', 'WinSvr08+', 'apple'
                 'mac_os_x', 'canonical',  'ubuntu_linux' , 'google',
                 'chrome_os', 'linux',  'linux_kernel', 'microsoft', 'internet_explorer',
                 'microsoft', 'windows_rt_8.1', 'microsoft', 'windows_vista']

    amEvenets = ['port 80',	'endpoint-malware', 'exploit_kit', 'fireeye_web_mps', 'fireeye_Hx', 'malicious-destination',
                 'malicious-email',	'malware', 'mcafee_vse', 'pup', 'ransomware', 'trojan', 'virus', 'windows 7']

    df_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    start_date = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2017-07-31', '%Y-%m-%d')
    for e in subevents:
        posts = ldap.getHackingPosts_Content()
        print(e, posts)
