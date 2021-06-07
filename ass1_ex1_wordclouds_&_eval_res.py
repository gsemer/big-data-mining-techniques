import pandas as pd
import matplotlib.pyplot as plt
# check the path for the files
import os
# create wordclouds
from wordcloud import WordCloud
# text cleaning
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# classification task
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

stop_words = stopwords.words('english')
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
for word in alphabet:
    if word not in stop_words:
        stop_words.append(word)
wnl = WordNetLemmatizer()


# make text lowercase
# remove text in square brackets
# remove punctuations
# remove URLS
# remove integers between words or symbols
def clean_text_round1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub("([^\x00-\x7F])+", ' ', text)
    return text


# tokenization step
def clean_text_round2(text):
    tokens = word_tokenize(text)
    token_words = ' '.join([word for word in tokens if not word in stop_words])
    return token_words


# lemmatization step
def clean_text_round3(text):
    tokens = word_tokenize(text)
    lemmatized = ' '.join([wnl.lemmatize(word) for word in tokens])
    return lemmatized


round1 = lambda x: clean_text_round1(x)
round2 = lambda x: clean_text_round2(x)
round3 = lambda x: clean_text_round3(x)


# main program starts here !!!
# the folder q1, which contains train.csv, test_without_labels.csv, has to be in the same folder as assignment_1.py
if os.path.isfile('datasets/q1/train.csv'):

    # Question 1a

    # create DataFrame for train.csv
    df = pd.read_csv("datasets/q1/train.csv", index_col=0)

    # preprocessing method for df
    for column in ['Content', 'Title']:
        df[column] = df[column].apply(round1)
        df[column] = df[column].apply(round2)

    # choose the rows depending on the category
    df_business = df[df['Label'] == 'Business']
    df_entertainment = df[df['Label'] == 'Entertainment']
    df_health = df[df['Label'] == 'Health']
    df_technology = df[df['Label'] == 'Technology']

    # create a cloud for each category
    cloud_business = WordCloud(width=600, height=300, max_font_size=50, max_words=100, background_color="white",
                               stopwords=stop_words).generate(str(df_business))
    cloud_entertainment = WordCloud(width=600, height=300, max_font_size=50, max_words=100, background_color="white",
                                    stopwords=stop_words).generate(str(df_entertainment))
    cloud_health = WordCloud(width=600, height=300, max_font_size=50, max_words=100, background_color="white",
                             stopwords=stop_words).generate(str(df_health))
    cloud_technology = WordCloud(width=600, height=300, max_font_size=50, max_words=100, background_color="white",

                                 stopwords=stop_words).generate(str(df_technology))
    # business's image
    plt.imshow(cloud_business, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # entertainment's image
    plt.imshow(cloud_entertainment, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # health's image
    plt.imshow(cloud_health, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # technology's image
    plt.imshow(cloud_technology, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Question 1b (Statistic Measure- without "My Method")

    # preprocessing method for df
    for column in ['Content']:
        df[column] = df[column].apply(round3)
    # method to remove duplicate rows based on all columns
    df.drop_duplicates(inplace=True)
    # consider the training data and target data
    X_train = df['Content']
    y_train = df['Label']
    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5)
    # BoW feature
    cv = CountVectorizer()
    X_train_bow = cv.fit_transform(X_train)
    # SVD feature
    svd = TruncatedSVD(n_components=100)
    X_train_svd = svd.fit_transform(X_train_bow)
    # Creating labels on target data
    label_encoder = preprocessing.LabelEncoder()
    X_train_label = label_encoder.fit_transform(y_train)
    # Support Vecton Machine and Random Forest
    svm = SVC(kernel='linear', gamma='auto', C=5)
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    # empty dictionaries
    accuracy = {}
    precision = {}
    recall = {}
    f_measure = {}
    # description of the columns of the output matrix
    classifier_feature = ['SVM BoW', 'RF BoW', 'SVM SVD', 'RF SVD']
    # evaluation step
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    for i in range(len(classifier_feature)):
        if classifier_feature[i] == 'SVM BoW':
            scores = cross_validate(svm, X_train_bow, X_train_label, scoring=scoring, cv=skf)
        if classifier_feature[i] == 'RF BoW':
            scores = cross_validate(rf, X_train_bow, X_train_label, scoring=scoring, cv=skf)
        if classifier_feature[i] == 'SVM SVD':
            scores = cross_validate(svm, X_train_svd, X_train_label, scoring=scoring, cv=skf)
        if classifier_feature[i] == 'RF SVD':
            scores = cross_validate(rf, X_train_svd, X_train_label, scoring=scoring, cv=skf)
        accuracy[classifier_feature[i]] = str(round(scores['test_accuracy'].mean(), 4))
        precision[classifier_feature[i]] = str(round(scores['test_precision_macro'].mean(), 4))
        recall[classifier_feature[i]] = str(round(scores['test_recall_macro'].mean(), 4))
        f_measure[classifier_feature[i]] = str(round(scores['test_f1_macro'].mean(), 4))
    # initialization of output matrix which it will contain the evaluation results
    evaluation_df = pd.DataFrame()
    evaluation_df.insert(loc=0, column='Statistic Measure', value=['Accuracy', 'Precision', 'Recall', 'F-Measure'])
    for i in range(len(classifier_feature)):
        evaluation_df.insert(loc=i + 1, column=classifier_feature[i], value=[accuracy[classifier_feature[i]],
                                                                             precision[classifier_feature[i]],
                                                                             recall[classifier_feature[i]],
                                                                             f_measure[classifier_feature[i]]])
    print(evaluation_df)
