import pandas as pd
from preprocess import NewsPreprocess,PricePreprocess
from extract_sentiment import Sentiment_Classifier
from scrap import create_csv
import os

if __name__=="__main__":
    if not os.path.isfile("../data/news_with_date.csv"):
        url="https://www.hydrocarbonprocessing.com/news?page="
        create_csv(url=url)
    data=pd.read_csv('../data/news_with_date.csv')
    news_data=NewsPreprocess(data)
    news_data=news_data.text_process()
    news_data.to_csv("../data/news_data.csv",index=False)
    
    wti_price_data=pd.read_csv("../data/wti_oil.csv")
    wti=PricePreprocess(wti_price_data)
    wti_price_data=wti.price_with_volume_and_date()
    wti_price_data.to_csv("../data/wti_data.csv",index=False)
    
    
    brent_price_data=pd.read_csv("../data/brent_oil.csv")
    brent=PricePreprocess(brent_price_data)
    brent_price_data=brent.price_with_volume_and_date()
    brent_price_data.to_csv("../data/brent_data.csv",index=False)
    
    data=pd.read_csv("../data/news_data.csv")
    df=Sentiment_Classifier(data)
    data_with_sentiment=df.sentiments_with_score_and_labels()
    data_with_sentiment.to_csv("../data/news_with_sentiment.csv",index=False)
