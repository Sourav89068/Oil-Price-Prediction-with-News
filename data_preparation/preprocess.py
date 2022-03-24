import pandas as pd
import nltk
import string
import re
english_stemmer=nltk.stem.SnowballStemmer('english')

class NewsPreprocess:
    def __init__(self,data: pd.core.frame.DataFrame):
        self.data=data
        
    def remove_punctuation(self,text: str)->str:
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_whitespace(self,text: str) -> str:
        return  " ".join(text.split())

    def date_format(self)-> pd.core.frame.DataFrame:
        data=self.data.copy()
        data['date'] = pd.to_datetime(data['date'],format="%m/%d/%Y")
        return data
    
    def text_process(self)-> pd.core.frame.DataFrame:
        data=self.date_format()
        data = data.groupby(['date'])['text'].apply(','.join).reset_index()
        data['text']=data['text'].apply(str.lower)
        data['text']=data['text'].apply(self.remove_punctuation)
        data['text']=data['text'].apply(self.remove_whitespace)
        return data
    
class PricePreprocess:
    def __init__(self,data: pd.core.frame.DataFrame):
        self.data=data
    
    def date_process(self,date_string : str) -> pd._libs.tslibs.timestamps:
        x=date_string
        month={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        new_date=x[4:6]+"-"+str(month[x[:3]])+"-"+x[8:13]
        return pd.to_datetime(pd.Series(new_date),format="%d-%m-%Y")
    
    def remove_comma(self,x: str)-> str:
        return x.replace(",","")
    
    def price_with_volume_and_date(self) -> pd.core.frame.DataFrame:
        data=self.data.copy()
        data['date']=data['Date'].apply(self.date_process)
        data=data.iloc[:,4:].drop(columns=["Adj Close**"])
        data.columns=["Close","Volume","date"]
        data=data[data.Close!="-"]
        data.Close=data.Close.astype("float64")
        data=data[data.Volume!="-"]
        data.Volume=data.Volume.apply(self.remove_comma)
        data.Volume=data.Volume.astype("float64")
        return data