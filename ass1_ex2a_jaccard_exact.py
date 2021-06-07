import pandas as pd 
import re 
import string
from nltk.corpus import stopwords
from tqdm import tqdm
import time


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
    text = re.sub('\n', ' ', text)
    if remove_stop_words:                                                   
        text = text.split()
        text = [w for w in text if w not in stop_words]
        text = ' '.join(text)
    return text


clean_function = lambda x: clean(x, False)


df = pd.read_csv('datasets/q2a/corpusTrain.csv')
test_df = pd.read_csv('datasets/q2a/corpusTest.csv')
df['Content'] = df['Content'].apply(clean_function)
test_df['Content'] = test_df['Content'].apply(clean_function)


# SET REPRESENTATION 


# df
set_dictionary_df = {} 
normal_dictionary_df = {} 
count_df = 1
for question in tqdm([x for x in df['Content']]):
   temporary_list = []
   for shingle in question.split(' '):
       if shingle not in stop_words:
           temporary_list.append(shingle.lower())
   set_dictionary_df['m{0}'.format(count_df)] = set(temporary_list)
   normal_dictionary_df['m{0}'.format(count_df)] = question
   count_df += 1


# test_df
set_dictionary_test_df = {} 
normal_dictionary_test_df = {} 
count_test_df = 1
for question in tqdm([x for x in test_df['Content']]):
   temporary_list = []
   for shingle in question.split(' '):
       if shingle not in stop_words:
           temporary_list.append(shingle.lower())
   set_dictionary_test_df['n{0}'.format(count_test_df)] = set(temporary_list)
   normal_dictionary_test_df['n{0}'.format(count_test_df)] = question
   count_test_df += 1


start_time = time.time()
duplicates = 0
for key_test_df in set_dictionary_test_df.keys():
    for key_df in set_dictionary_df.keys():
        a = len(set_dictionary_test_df[key_test_df].intersection(set_dictionary_df[key_df]))
        b = len(set_dictionary_test_df[key_test_df].union(set_dictionary_df[key_df]))
        result = round(a/b, 3)
        if result >= 0.8:
            duplicates += 1
            break 
elapsed_time = time.time() - start_time
print(elapsed_time)
print(duplicates)

