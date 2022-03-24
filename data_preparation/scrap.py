from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

def create_csv(url: str)->None:
    dates=[]
    texts=[]
    for i in range(1,641):
        url=url
        res = requests.get(url+str(i))
        rt=BeautifulSoup(res.content,"html.parser")
        length=len(rt.find_all("div",class_="news-link"))
        for j in range(length):
            date=rt.find_all("div",class_="news-link")[j].find("span").get_text()
            text=rt.find_all("div",class_="news-link")[j].find("a").get_text()
            dates.append(date)
            texts.append(text)
    pd.DataFrame(list(zip(dates,texts)),columns=["date","text"]).to_csv("data/news_with_date.csv",index=False)
    return
