import pandas as pd
import numpy as np
import seaborn as sns
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
df = pd.read_csv("emotion_dataset_raw.csv")
print(df.head())
print(df['Emotion'].value_counts())
print(dir(nfx))
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
print(df)
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(max_iter=1000))])
pipe_lr.fit(x_train,y_train)
print(pipe_lr)
print(pipe_lr.score(x_test,y_test))
ex1 = "This book was so interesting it made me happy"
print(pipe_lr.predict([ex1]))
print(pipe_lr.classes_)
import joblib
pipeline_file = open("emotion_classifier_pipe_lr.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl", "rb"))
print(pipe_lr.predict([ex1]))