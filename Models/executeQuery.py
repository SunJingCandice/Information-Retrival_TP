
# coding: utf-8

# In[ ]:


#Initialise the query execution by storing the information needed for query execution:
#@param documents: the representaiton of the documents of the corpus as scipy.sparse.csr.csr_matrix
#@param documentIDs: list (or array) containing the IDs of the documents in the same order as documents.
#@param translationFunction: function which takes the query as argument and returns the representation.
#    Can be implemented by handling the query as document, so it will return a document vector.
#    @see https://www.filipekberg.se/2009/11/30/function-pointers-in-c-and-python-and-their-possible-usage/
#@author: Thorsten
import sklearn.preprocessing
import scipy
import numpy as np
#import tensorflow as tf #was slower in the tests
docs=None
func=None
docIDs=None
#sess = tf.InteractiveSession()
def initExecution(documents, documentIDs, transformationFunction):
    global docs
    global func
    global docIDs
    docs=sklearn.preprocessing.normalize(documents, norm='l2', copy=(type(documents)!=np.ndarray))
    #docs=tf.constant(np.array(sklearn.preprocessing.normalize(documents, norm='l2', copy=(type(documents)!=np.ndarray)).toarray()))
    func=transformationFunction
    docIDs=documentIDs


# In[ ]:


#Execute the given query on the data set and using the method given before using the method initExecution.
#@param query: string, the query that should be executed
#@param threshold: float, all documents better or equal rated than this threshold will be returned as results.
#    Default: 0.0. The results will have ratings in [-1,1], so set to -1 to get all documents.
#@param sort: boolean, define if the results should be sorted descending (True). Can be disabled for saving
#    the computational power for sorting. Default: True (enabled)
#@return results: numpy.array, containing the results of the query (document IDs).
#    Note: Results rated worse than the given threshold will not be contained.
#@return similarities: numpy.array, the similarities for the results
#@author: Thorsten
def executeQuery(query, threshold=-1.0, sort=True):
    #if val is not None
    if((docs is None)|(func is None)|(docIDs is None)): #test if initialised
        print('Error: Tryed to execute query without initialising.')
        print('Call the method initExecution(documents, transformationFunction) for preventing the needed information.')
        return None
    queryRepresentation=func(query).toarray()[0]
    #normalize
    s=np.linalg.norm(queryRepresentation)
    queryRepresentation=queryRepresentation if s==0 else queryRepresentation/s
    #compute the similarity (here cosine similarity)
    #dot product is enough becuase it was normalized before
    
    #tensorflow version
    #b=tf.constant(queryRepresentation, dtype=tf.float64)
    #similarities=tf.einsum('dw,w->d', docs, b).eval()
    
    similarities=np.array(list(map(queryRepresentation.dot, docs.toarray())))

    #old code, no longer needed
    #similarities=[]
    #results=[]
    #for i in range(docs.shape[0]):
    #    sim=np.dot(docs[i].toarray(), queryRepresentation)[0] #enough becuase normalized before
    #    if(sim>=threshold):
    #        similarities.append(sim)
    #        results.append(docIDs[i])

    if(type(docIDs)!=np.ndarray):
        results=np.copy(docIDs.values)
    else:
        results=np.copy(docIDs)
    
    #apply threshold
    i=0
    while(i<len(similarities)):
        if similarities[i]<threshold:
            results=np.delete(results, i)
            similarities=np.delete(similarities, i)
        else:
            i+=1
    
    #sort the result
    similarities=np.array(similarities)
    results=np.array(results)
    if(sort==True):
        inds=np.argsort(similarities)[::-1]
        similarities=similarities[inds]
        results=results[inds]
    
    return results, similarities


# In[ ]:


#for testing
#import DTM_generate as generateDTM
#import LDA_Topic_Model as generateDTM
#initExecution(generateDTM.train_tfidf, generateDTM.train_doc_file['id'], generateDTM.get_QueryVector)
#content of document MED-301
#query=['methylmercury potential environmental risk factor contributing epileptogenesis abstract epilepsy seizure disorder common neurological diseases humans genetic mutations ion channels receptors risk factors brain injury linked epileptogenesis underlying majority epilepsy cases remains unknown gene-environment interactions thought play critical role etiology epilepsy exposure environmental chemicals important risk factor methylmercury mehg prominent environmental neurotoxicant targets primarily central nervous system cns patients animals acute chronic mehg poisoning display epileptic seizures show increased susceptibility seizures suggesting mehg exposure epileptogenesis mini-review highlights effects mehg exposure developmental exposure susceptibility humans animals seizures discusses potential role low level mehg exposure epileptogenesis review proposes preferential effect mehg inhibitory gabaergic system leading disinhibition excitatory glutamatergic function potential mechanisms underlying mehg-induced seizure susceptibility']
#query=(['This is a clinical test!'])
#result=executeQuery(query)#, 1e-5, sort=True)
#print()
#result=executeQuery(query)
#print(len(result[0]))
#print(result[0])
#print(result[1])
#print(max(result[1]))
#print(min(result[1]))
#print()
#for i in range(len(result[0])):
#    if(result[0][i]=='MED-301'):
#        print('Found: ', i)
#        break
##print(docIDs)


# In[ ]:


##for i in range(len(generateDTM.train_doc_file['id'])):
##    e=generateDTM.train_doc_file['id'][i]
##    print(i, e)
##    if e=='MED-301':
##        print(i)
##        break
##i=3
#
#vecDoc=generateDTM.get_QueryVector([generateDTM.train_doc_file['text'][3]]).toarray()[0]
#
#doc2=docs.toarray()[3]
#print(len(doc2))
#print(doc2[0])
#print()
#
#vecQuery=generateDTM.get_QueryVector(query).toarray()[0]
#
#queryRepresentation=func(query).toarray()[0]
##normalize
#s=np.linalg.norm(queryRepresentation)
#queryRepresentation=queryRepresentation if s==0 else queryRepresentation/s
#
#vecDoc=vecDoc/np.linalg.norm(vecDoc)
#
#print(queryRepresentation.dot(vecDoc))
##r=list(map(queryRepresentation.dot, [docs.toarray()[3]]))
##print(max(r))
##print(r)
#
#print()
#print()
#
#print(queryRepresentation.dot(doc2))
##r=r=list(map(queryRepresentation.dot, [doc2]))
##print(max(r))
##print(r)
#
##print(np.dot(vecDoc, vecQuery)/(np.linalg.norm(vecDoc)*np.linalg.norm(vecQuery)))


# In[ ]:


#import DTM_generate
##import LDA_Topic_Model as DTM_generate
##import IRByRandomLeaderPreClustering as DTM_generate
#import numpy as np
#print()
#print()
#
#doc1=DTM_generate.train_doc_file['text'][3]
#doc1=DTM_generate.get_QueryVector([doc1]).toarray()[0]
#
#doc2=DTM_generate.train_tfidf.toarray()[3]
##now both should contain the same entries; once generated by the method, once taken from the stored matix
#
#print('np.linalg.norm(doc1-doc2): ', np.linalg.norm(doc1-doc2), '(shuold be 0 or very close to it)')
##maybe different lengths becasue it could be normalized when learned?
#
#doc1=doc1/np.linalg.norm(doc1)
#doc2=doc2/np.linalg.norm(doc2)
#print('After normalizing:\nnp.linalg.norm(doc1-doc2): ', np.linalg.norm(doc1-doc2), '(shuold be 0 or very close to it)')
##better because closer to 0, but still not 0.
#print('np.dot(doc1, doc2): ', np.dot(doc1, doc2), '(ideally 1, but high is also OK becuase doc2 has IDF and doc 1 not)')


# In[ ]:


#t=generateDTM.train_tfidf
#vecDoc=t[3].shape
#print(type(vecDoc))
#vecDoc=vecDoc/np.linalg.norm(vecDoc)
#print(len(vecDoc))
#print(np.dot(vecDoc, vecQuery))


# In[ ]:


#import time
#import DTM_generate
#initExecution(DTM_generate.train_tfidf, DTM_generate.train_doc_file['id'], DTM_generate.get_QueryVector)
#query=['This is a test!']
#query=['how animal proteins may trigger autoimmune disease']
#timeBefore=time.perf_counter()
#DTM_generate.get_QueryVector(query)
#timeAfter=time.perf_counter()
#print('Time for converting:\t\t', timeAfter-timeBefore)
#timeBefore2=time.perf_counter()
#executeQuery(query)
#timeAfter2=time.perf_counter()
#print('Time for the execution itself:\t', timeAfter2-timeBefore2-timeAfter+timeBefore)
#print('Time in total:\t\t\t', timeAfter2-timeBefore2)


# In[ ]:


#def function(a):
#    return a if type(a)==np.ndarray else np.array(a)    
#A = np.array(
#[[0, 1, 0, 0, 1],
#[0, 0, 1, 1, 1],
#[1, 1, 0, 1, 0]])
#IDs=np.array(['document 1', 'document 2', 'document 3'])
#initExecution(A, IDs, function)


# In[ ]:


#query=[0.1,10,3, 4, -8.7]
#query=[1,10.1,15, -20, 4.8]
#print(executeQuery(query, -1, False))
#print(executeQuery(np.array(query)))

