from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

class Sentiment_Classifier:
    def __init__(self,data):
        self.data=data
        
    def sentiment_scores(self,text):
        inputs=tokenizer(text,return_tensors="pt",padding=True)
        outputs=finbert(**inputs)[0]
        sentiment_array=outputs.detach().numpy()[0]
        return sentiment_array
    
    def sentiments_from_scores(self,score_array):
        labels = {0:'neutral', 1:'positive',2:'negative'}
        label=labels[np.argmax(score_array)]
        return label
    def sentiments_with_score_and_labels(self):
        data=self.data.copy()
        sentiment_scores=data['text'].apply(self.sentiment_scores)
        sentiment_labels=sentiment_scores.apply(self.sentiments_from_scores)
        sentiment_score_dataframe=pd.DataFrame(sentiment_scores.to_list(),columns=["neutral_score","positive_score","negative_score"])
        sentiment_labels_dataframe=pd.DataFrame(sentiment_labels)
        sentiment_labels_dataframe.columns=["sentiment"]
        data_with_sentiment=pd.concat([data[["text","date"]],sentiment_score_dataframe,sentiment_labels_dataframe],axis=1)
        return data_with_sentiment