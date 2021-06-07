import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class Methods():

    def __init__(self, df):
        self.df = df
        self.stop_words = stopwords.words('english')
        self.clean_function = lambda x: self.clean(x, False)
        for column in ['Question1', 'Question2']:
            self.df[column] = self.df[column].apply(self.clean_function)
        self.df['concat_questions'] = self.df['Question1']+' '+self.df['Question2']
        self.cross_validation()
        self.X_train = self.df['concat_questions']
        self.y = self.df['IsDuplicate']
        self.feature_tfidf()
        self.feature_bow()
        self.model_xgb()
        self.model_rf()
        self.statistic_measure()

    def clean(self, text, remove_stop_words):
        text = str(text)
        text = text.lower()
        text = re.sub('whats', 'what is', text)
        text = re.sub("\'ve", ' have', text)
        text = re.sub("\'ll", ' will', text)
        text = re.sub("i'm", 'i am', text)
        text = re.sub("can't", 'can not', text)
        text = re.sub("\'re", ' are', text)
        text = re.sub("\'d", ' would', text)
        text = re.sub("n't", ' not', text)
        text = re.sub('\$', 'dollar', text)
        text = re.sub('\&', 'and', text)
        text = re.sub('\%', 'percent', text)
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub("([^\x00-\x7F])+", ' ', text)
        text = re.sub(re.compile('<.*?>'), ' ', text)
        text = re.sub('\w*\d\w*', ' ', text)
        text = re.sub("\'s", ' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        if remove_stop_words:
            text = text.split()
            text = [w for w in text if w not in self.stop_words]
            text = ' '.join(text)
        return text
    
    def cross_validation(self):
        self.skf = StratifiedKFold(n_splits=5)
   
    def feature_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.X_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
      
    def feature_bow(self):
        self.bow_vectorizer = CountVectorizer()
        self.X_bow = self.bow_vectorizer.fit_transform(self.X_train)
                         
    def model_xgb(self):
        self.xgb = xgb.XGBClassifier(n_estimators=100)
        
    def model_rf(self):
        self.rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
        
    def statistic_measure(self):
        self.precision, self.recall, self.f_measure, self.accuracy = ({} for i in range(4))
        self.scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        self.methods = ['Method-1', 'Method-2']
        self.evaluation_df = pd.DataFrame()
        for i in range(len(self.methods)):
            print('Evaluation results - {}'.format(self.methods[i]))
            if self.methods[i] == 'Method-1':
                self.scores = cross_validate(self.rf, self.X_tfidf, self.y, scoring=self.scoring, cv=self.skf)
            if self.methods[i] == 'Method-2':
                self.scores = cross_validate(self.xgb, self.X_bow, self.y, scoring=self.scoring, cv=self.skf)
            print('Accuracy: {}'.format(str(self.scores['test_accuracy'])))
            print('Precision: {}'.format(str(self.scores['test_precision_macro'])))
            print('Recall: {}'.format(str(self.scores['test_recall_macro'])))
            print('F-Measure: {}'.format(str(self.scores['test_f1_macro'])))
            self.accuracy[self.methods[i]] = str(round(self.scores['test_accuracy'].mean(), 4))
            self.precision[self.methods[i]] = str(round(self.scores['test_precision_macro'].mean(), 4))
            self.recall[self.methods[i]] = str(round(self.scores['test_recall_macro'].mean(), 4))
            self.f_measure[self.methods[i]] = str(round(self.scores['test_f1_macro'].mean(), 4))
        self.evaluation_df.insert(loc=0, column='Statistic Measure', value=['Accuracy', 'Precision', 'Recall', 'F-Measure'])
        for i in range(len(self.methods)):
            self.evaluation_df.insert(loc=i+1, column=self.methods[i], value=[self.accuracy[self.methods[i]],
                                                                              self.precision[self.methods[i]],
                                                                              self.recall[self.methods[i]],
                                                                              self.f_measure[self.methods[i]]])
        print(self.evaluation_df)

if __name__ == '__main__':
    dataframe = pd.read_csv('datasets/q2b/train.csv')
    print(Methods(dataframe))

