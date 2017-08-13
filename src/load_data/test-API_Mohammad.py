
from crawler import *
import pandas as pd



def text_API(url = 'https://apigargoyle.com/GargoyleApi/getDetailedVulnerabilityInfo?limit=3000'):
    print url
    headers = {"userId" : "labuser", "apiKey" : "a9a2370f-4959-4511-b263-5477d31329cf"}
    response = requests.get(url, headers=headers)

    return response.json()['results']





def main():
    results = text_API("https://apigargoyle.com/GargoyleApi/getDetailedVulnerabilityInfo?limit=2500")
    #text_API(url='https://apigargoyle.com/GargoyleApi/generateForumMarketNet?limit=5000')
    print len(results)
    df = pd.DataFrame.from_dict(results)
    for i in results:
        print i
        pass


    print len(results)
if __name__ == "__main__":
    main()


"""


            #write the product and the financial institute tag to a file

"""
