import pandas as pd
import re 
import string
from nltk.corpus import stopwords
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH


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
   
   
permutations = [16, 32, 64]


for i in range(len(permutations)):
    
    num_perm = permutations[i]
    
    print('Create MinHash signatures for {} number of permutations on train set'.format(str(num_perm)))
    min_dictionary_df = {} 
    count_df = 1
    for value in tqdm(set_dictionary_df.values()):
        m = MinHash(num_perm=num_perm)
        for shingle in value:
            m.update(shingle.encode('utf8'))
        min_dictionary_df['m{}'.format(count_df)] = m
        count_df += 1
   
    print('Create MinHash signatures for {} number of permutations on test set'.format(str(num_perm)))
    min_dictionary_test_df = {}
    count_test_df = 1
    for value in tqdm(set_dictionary_test_df.values()):
        m = MinHash(num_perm=num_perm)
        for shingle in value:
             m.update(shingle.encode('utf8'))
        min_dictionary_test_df['n{}'.format(count_test_df)] = m
        count_test_df += 1
   
    print('Create LSH index for {} number of permutations'.format(str(num_perm)))
    lsh = MinHashLSH(threshold=0.8, num_perm=num_perm)
    for key in tqdm(min_dictionary_df.keys()):
        lsh.insert(key, min_dictionary_df[key])
   
    big_list = []
    for query in min_dictionary_test_df.keys():
        big_list.append(lsh.query(min_dictionary_test_df[query]))
   
    duplicates = 0
    for j in range(len(big_list)):
        if len(big_list[j]) != 0:
            duplicates += 1
    print(duplicates)

