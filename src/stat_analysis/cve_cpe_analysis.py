import pandas as pd
import pickle


if __name__ == "__main__":
    vulnInfo = pickle.load(open('../../data/DW_data/Vulnerabilities-sample_v2+.pickle', 'rb'))
    print(vulnInfo)