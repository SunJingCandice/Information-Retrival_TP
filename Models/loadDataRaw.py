
# coding: utf-8

# In[ ]:


#Import the data of the given file.
#@author: Thorsten


# In[ ]:


#Import the data of the file specifiedin filename.
#@author
#@param filename: string, the filename of the file that should be opened.
#    Can also be a relative path.
#@param separator: character (or string) for the separation of the entries of the file.
#    Default is \t (tabulator), because it's used in our data set.
#@return data: list where each entry is one line in the given file. Each entry has as many dimensions,
#    as the line in the file had (separated by the character in separator).
def readFile(filename, separator='\t'):
    data=[]
    #print("Start reading the data.")
    with open(filename) as file:
        for line in file:
            data.append(line.strip().split(separator))
    #print("Completed reading the data.")
    return data

#for testing
#data=readFile('nfcorpus/raw/nfdump.txt')
#data=readFile('nfcorpus/raw/doc_dump.txt')
#print(len(data))
#print(len(data[0]))
#for x in data[0]:
#    print(x)ids=readFile('nfcorpus/raw/train.docs.ids')
#ids=readFile('nfcorpus/raw/dev.docs.ids')
#ids=readFile('nfcorpus/raw/test.docs.ids')

#print(len(ids))
#print(len(ids[0]))
#for x in ids[0]:
#    print(x)train=[]
#for id in ids:
#    for item in data:
#        if(id[0]==item[0]):
#            train.append(item)

#faster way
#pointer=0
#data.sort()
#ids.sort()
#for id in ids:
#    while(data[pointer][0] != id[0]):
#        pointer+=1
#    train.append(data[pointer])
#print(len(train))
