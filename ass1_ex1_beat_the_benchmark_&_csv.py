import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate


class Predict():

    def __init__(self, df, test_df):
        self.df = df
        self.test_df = test_df
        self.stop_words = stopwords.words('english')
        self.wnl = WordNetLemmatizer()
        self.clean_function()
        # Consider train, test data
        self.X_train = self.df['Content']
        self.y_train = self.df['Label']
        self.X_test = self.test_df['Content']
        # 5-Fold cross validation
        self.skf = StratifiedKFold(n_splits=5)
        # Convert X_train, X_test into vectors according to tf-idf vectorizer
        self.tfidf = TfidfVectorizer()
        self.X_train_tfidf = self.tfidf.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf.transform(self.X_test)
        # Create labels on target data ( y_train )
        self.label_encoder = preprocessing.LabelEncoder()
        self.X_train_label = self.label_encoder.fit_transform(self.y_train)
        # Multinomial Naive Bayes
        self.model = MultinomialNB(alpha=0.02, fit_prior=True)
        self.model.fit(self.X_train_tfidf, self.X_train_label)
        # Statistic Measure
        self.statistic_measure()
        # Prediction
        self.prediction_function()
        # New dataframe
        self.create_dataframe()
        # Create csv file
        self.create_csv()

    def clean_function(self):
        for column in ['Content']:
            self.df[column] = self.df[column].apply(lambda x: self.clean_text_round1(x))
            self.df[column] = self.df[column].apply(lambda x: self.clean_text_round2(x))
            self.df[column] = self.df[column].apply(lambda x: self.clean_text_round3(x))
            self.test_df[column] = self.test_df[column].apply(lambda x: self.clean_text_round1(x))
            self.test_df[column] = self.test_df[column].apply(lambda x: self.clean_text_round2(x))
            self.test_df[column] = self.test_df[column].apply(lambda x: self.clean_text_round3(x))
        print(self.df['Content'])
        print(self.test_df['Content'])

    def clean_text_round1(self, text):
        text = text.lower()
        text = re.sub('\[.*?\]', ' ', text)
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub('<.*?>+', ' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('\w*\d\w*', ' ', text)
        return text

    def clean_text_round2(self, text):
        self.tokens = word_tokenize(text)
        self.token_words = ' '.join([word for word in self.tokens if not word in self.stop_words])
        return self.token_words

    def clean_text_round3(self, text):
        self.tokens = word_tokenize(text)
        self.lemmatized = ' '.join([self.wnl.lemmatize(word) for word in self.tokens])
        return self.lemmatized

    def statistic_measure(self):
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.f_measure = {}
        self.classifier_feature = ['My Method']
        self.scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for i in range(len(self.classifier_feature)):
            if self.classifier_feature[i] == 'My Method':
                self.scores = cross_validate(self.model, self.X_train_tfidf, self.X_train_label, scoring=self.scoring, cv=self.skf)
            print(self.scores)
            self.accuracy[self.classifier_feature[i]] = str(round(self.scores['test_accuracy'].mean(), 4))
            self.precision[self.classifier_feature[i]] = str(round(self.scores['test_precision_macro'].mean(), 4))
            self.recall[self.classifier_feature[i]] = str(round(self.scores['test_recall_macro'].mean(), 4))
            self.f_measure[self.classifier_feature[i]] = str(round(self.scores['test_f1_macro'].mean(), 4))
        self.new_df = pd.DataFrame()
        self.new_df.insert(loc=0, column='Statistic Measure', value=['Accuracy', 'Precision', 'Recall', 'F-Measure'])
        for i in range(len(self.classifier_feature)):
            self.new_df.insert(loc=i+1, column=self.classifier_feature[i], value=[self.accuracy[self.classifier_feature[i]],
                                                                                  self.precision[self.classifier_feature[i]],
                                                                                  self.recall[self.classifier_feature[i]],
                                                                                  self.f_measure[self.classifier_feature[i]]])
        print(self.new_df)

    def prediction_function(self):
        self.predictions_label = self.model.predict(self.X_test_tfidf)
        self.predictions = self.label_encoder.inverse_transform(self.predictions_label)
        return self.predictions

    def create_dataframe(self):
        self.final_df = pd.DataFrame({'Id': self.test_df['Id'], 'Predicted': self.predictions})
        return self.final_df

    def create_csv(self):
        self.final_df.to_csv('testSet_categories.csv', header=True, index=False)


if __name__ == '__main__':
    if os.path.isfile('datasets/q1/train.csv') and os.path.isfile('datasets/q1/test_without_labels.csv'):
        train_dataframe = pd.read_csv('datasets/q1/train.csv')
        test_dataframe = pd.read_csv('datasets/q1/test_without_labels.csv')
        print(Predict(train_dataframe, test_dataframe))
    else:
        print('Put folder datasets and file ass1_ex1_beat_the_benchmarl_&_csv.py in the same folder')

