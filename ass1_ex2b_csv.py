import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

stop_words = stopwords.words('english')
def clean(text, remove_stop_words):
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
        text = [w for w in text if w not in stop_words]
        text = ' '.join(text)
    return text
clean_function = lambda x: clean(x, False)

df = pd.read_csv('datasets/q2b/train.csv')
test_df = pd.read_csv('datasets/q2b/test_without_labels.csv')

for column in ['Question1', 'Question2']:
    df[column] = df[column].apply(clean_function)
    test_df[column] = test_df[column].apply(clean_function)

df['concat_questions'] = df['Question1']+' '+df['Question2']
test_df['concat_questions'] = test_df['Question1']+' '+test_df['Question2']

X_train = df['concat_questions']
X_test = test_df['concat_questions']
y = df['IsDuplicate']

label_encodel = LabelEncoder()
X_1_label = label_encodel.fit_transform(y)

cv = TfidfVectorizer()

X_1 = cv.fit_transform(X_train)
X_2 = cv.transform(X_test)

forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_1, X_1_label)

predictions_label = forest.predict(X_2)
predictions = label_encodel.inverse_transform(predictions_label)

predict_df = pd.DataFrame({'Id': test_df['Id'], 'Predicted': predictions})
print(predict_df)

predict_df.to_csv('duplicate_predictions.csv', header=True, index=False)

