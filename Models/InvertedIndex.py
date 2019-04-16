
# coding: utf-8

# In[ ]:


#This file is made to create and use an inverted index.
#@author: Thorsten
invertedIndex=None #the inverted index, dictionary of lists
index=None # the normal index, matrix[doc][word]=count, line-normalized
wordList=None #the list of words in the order of the columns of the index = order of rows of inverted Index


# In[ ]:


#Preprocess text, e. g. the queries as first step of their execution,
#but also by the functino buildIndex.
#@param text: Strings, the text that should be preprocessed.
#@return List of strings: the preprocessed input list, splitted to words
#@author: Thorsten, parts taken from Hailian's LDA_Topic_Model.

# import stopword and punctuation lists, taken from Hailian
import nltk
from nltk.corpus import stopwords
from string import punctuation
nltk.download("stopwords")
esw = stopwords.words("english")

#prepare removing the punctuation
outtab=''
for i in punctuation:
    outtab+=' '
trantab = str.maketrans(punctuation, outtab)

ps=nltk.stem.PorterStemmer()

def preprocess(text):
    text=text.lower()
    #remove punctuation by replacing with spaces
    #If just removed, words might merge.
    text=text.translate(trantab)
    #remove double spaces
    while('  ' in text):
        text=text.replace('  ', ' ')
    text=text.split(' ')
    i=0
    while(i<len(text)): #trick because empty strings will be deleted
        #remove stopwords
        if(text[i] in esw):
            del text[i]
            continue
        text[i]=ps.stem(text[i]).strip()
        if(text[i]==''):
            del text[i]
            continue
        i+=1
    return text


# In[ ]:


#Build the index using the given documents.
#@param docs: List of list strings, the content of the documents which should be indexed, after preprocessing,
#    so every outer list contains a list of strings which are one document.
#@param ids: List, the IDs of the documents in the same order as in docs.
#    Used later for identifying the documents indexed.
#@author: Thorsten
import numpy as np
def buildIndex(docs, ids):
    global invertedIndex
    global index
    global wordList
    #get all words, but every word just one time
    wordList=set()
    for document in docs:
        for word in document:
            wordList.add(word)
    wordList=np.sort(np.array(list(wordList)))
    #build the index
    index=np.zeros([len(docs), len(wordList)])
    for i in range(len(docs)):
        docs[i]=np.sort(docs[i])
        pointer=0
        for j in range(len(docs[i])):
            while(wordList[pointer]<docs[i][j]):
                pointer+=1
            index[i][pointer]+=1
    #build the inverted index
    invertedIndex={}
    for word in wordList:
        invertedIndex[word]=[]
    for i in range(len(docs)):
        for k in invertedIndex.keys():
            if k in docs[i]:
                invertedIndex[k].append(ids[i])
    #Till now the index only contains TF (term frequency).
    #Now let's add IDF (inverdet document frequency)
    for j in range(len(wordList)):
        index[:,j]/=len(invertedIndex[wordList[j]])
    #normalize
    for i in range(len(docs)):
        index[i]/=np.linalg.norm(index[i])


# In[ ]:


#for testing
#import time
#teststring='This is a test-string!            Just play/trick. asdf he is, she was , dump, dumps, trains, simulations.'
#docs="This is just a simple test document. Try and let's see what happens. Just test, test and test."
#dump='test test test test test'
#teststring=preprocess(teststring)
#docs=preprocess(docs)
#dump=preprocess(dump)
#timeBefore=time.perf_counter()
#buildIndex([teststring, docs, dump], [0, 1, 2])
#timeAfter=time.perf_counter()
#print(timeAfter-timeBefore)
#print(invertedIndex)


# In[ ]:


#hardcore test
#import loadDataRaw
#import time
#documents=np.array(loadDataRaw.readFile('train.docs'))
#ids=documents[:,0]
#documents=documents[:,1]
#timeBefore=time.perf_counter()
#documents=[preprocess(doc) for doc in documents]
#timeAfter=time.perf_counter()
#print('Time for preprocessing: ', timeAfter-timeBefore)


# In[ ]:


#timeBefore=time.perf_counter()
#buildIndex(documents, ids)
#timeAfter=time.perf_counter()
#print('Time for index building: ', timeAfter-timeBefore)

