
# coding: utf-8

# In[ ]:


#This file is responsible for executing a single query on the inverted index.
#Builds on executeQuery.
#@author: Thorsten


# In[ ]:


#Initialise the query execution by storing the information needed for query execution.
#Also calls the build index function, so initializes the inverted index.
#@param documents: documents as list of strings, each string is one document.
#@param documentIDs: list (or array) containing the IDs of the documents in the same order as documents.
#@param translationFunction: function which takes the query as argument and does the preprocessing, necessary to
#    execute the query. Here returns a list of strings.
#    @see https://www.filipekberg.se/2009/11/30/function-pointers-in-c-and-python-and-their-possible-usage/
#@param initFunction: function which builds/initializes the index. Docs will be transformed using the
#    given transformation function before passed through.
#@author: Thorsten

import InvertedIndex
import numpy as np
func=None
docIDs=None
invertedIndex=None
index=None
wordList=None
docIDs_LookupTable=None
def initExecution(documents, documentIDs, transformationFunction=InvertedIndex.preprocess, initFunction=InvertedIndex.buildIndex):
    global func
    global docIDs
    global invertedIndex
    global index
    global wordList
    global docIDs_LookupTable
    func=transformationFunction
    docIDs=np.array(documentIDs)
    documents=np.array(documents)
    documents=[func(doc) for doc in documents]
    InvertedIndex.buildIndex(documents, docIDs)
    invertedIndex=InvertedIndex.invertedIndex
    index=InvertedIndex.index
    wordList=InvertedIndex.wordList
    docIDs_LookupTable={}
    for i in range(len(docIDs)):
        docIDs_LookupTable[docIDs[i]]=i


# In[ ]:


#Execute the given query on the data set and using the method given before using the method initExecution.
#First it's tryed to execute the query using the inverted index. If documents survive, they are ranked
#using the cosine similarity which is computed only for these documents to the query.
#If no documnets survive the inverted index, this method falls back to executing the query in the normal way;
#computing the cosine similiarity to all documents in the collection.
#@param query: string, the query that should be executed
#@param sort: boolean, define if the results should be sorted descending (True). Can be disabled for saving
#    the computational power for sorting. Default: True (enabled)
#@return results: numpy.array, containing the results of the query (document IDs).
#    Note: Not all documents in the collection are returned if the inverted index search was good..
#@return similarities: numpy.array, the similarities for the results, same order.
#@author: Thorsten
import functools
#import tensorflow as tf #in the tests, tensorflow was slower
#sess = tf.InteractiveSession()
def executeQuery(query, sort=True):
    #if val is not None
    if((func is None)|(docIDs is None)|(invertedIndex is None)|(index is None)|(wordList is None)): #test if initialised
        print('Error: Tryed to execute query without initialising.')
        print('Call the method initExecution(documents, transformationFunction) for preventing the needed information.')
        return None
    #preprocess the query
    queryRepresentation=func(query)
    if(len(queryRepresentation)<1):
        return np.array([]), np.array([]) #all search words removed as stop words which are not indexed
    #first try by using the inverted index
    indexLists=np.array([invertedIndex.get(word, []) for word in queryRepresentation])
    #merge the lists
    indexLists=functools.reduce(np.intersect1d, indexLists)
    if(len(indexLists)==0): #inverted index search found nothing which contains all words
        indexLists=np.array(range(len(index))) #work with all index entries from now on
    else:
        #indexLists=np.array([list(docIDs).index(word) for word in indexLists]) #too slow
        indexLists=np.array([docIDs_LookupTable[word] for word in indexLists])
    #get the word vector
    queryRepresentation=np.sort(queryRepresentation)
    wordVector=np.zeros(len(wordList))
    pointer=0
    for i in range(len(queryRepresentation)):
        if(not queryRepresentation[i] in wordList):
            continue
        while(wordList[pointer]<queryRepresentation[i]):
            pointer+=1
        wordVector[pointer]+=1
    norm=np.linalg.norm(wordVector)
    if(norm==0):
        #no word of the query indexed, so nothing found
        return np.array([]), np.array([])
    wordVector/=norm #normalize the word vector, doc vectors are already
    #compute the similarity (here cosine similarity)
    #dot product is enough becuase it was normalized before
    similarities=np.array(list(map(wordVector.dot, index[indexLists])))
    #a=tf.constant(index[indexLists])
    #b=tf.constant(wordVector)
    #similarities=tf.einsum('dw,w->d', a, b).eval()

    results=docIDs[indexLists]
    #sort the result
    if(sort==True):
        inds=np.argsort(similarities)[::-1]
        similarities=similarities[inds]
        results=results[inds]
    return results, similarities


# In[ ]:


#for testing
#teststring='This is a test-string!            Just play/trick. asdf he is, she was , dump, dumps, trains, simulations.'
#docs="This is just a simple test document. Try and let's see what happens. Just test, test and test."
#dump='test test test test test'
#initExecution([teststring, docs, dump], ['teststring', 'docs', 'dump'])


# In[ ]:


#query='Simple test!'
#results, similarities=executeQuery(query)
#print(results)
#print(similarities)


# In[ ]:


#import loadDataRaw
#import time
#documents=np.array(loadDataRaw.readFile('train.docs'))
#initExecution(documents[:,1], documents[:,0])


# In[ ]:


#query='This is a test!'
#query='This is a tset shit!'
#query='statin breast cancer survival nationwide cohort study finland abstract recent studies suggested statins established drug group prevention cardiovascular mortality delay prevent breast cancer recurrence effect disease-specific mortality remains unclear evaluated risk breast cancer death statin users population-based cohort breast cancer patients study cohort included newly diagnosed breast cancer patients finland num num num cases identified finnish cancer registry information statin diagnosis obtained national prescription database cox proportional hazards regression method estimate mortality statin users statin time-dependent variable total num participants statins median follow-up num years diagnosis range num num years num participants died num num due breast cancer adjustment age tumor characteristics treatment selection post-diagnostic pre-diagnostic statin lowered risk breast cancer death hr num num ci num num hr num num ci num num risk decrease post-diagnostic statin affected healthy adherer bias greater likelihood dying cancer patients discontinue statin association dose-dependent observed low-dose/short-term dose time-dependence survival benefit pre-diagnostic statin users suggests causal effect evaluated clinical trial testing statins effect survival breast cancer patients '
#timeBefore=time.perf_counter()
#result=executeQuery(query, sort=False)
#timeAfter=time.perf_counter()
#print(timeAfter-timeBefore)
#print(len(result[1]))
#print(InvertedIndex.preprocess(query))
#print(result[1])

