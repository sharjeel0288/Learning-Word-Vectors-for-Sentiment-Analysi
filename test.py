import os
import numpy as NP
from scipy.special import logsumexp
import re
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import random
import json
import matplotlib.pyplot as plt
import time
vectors = NP.load('../train/vectors1.txt.npy') # load
labels = NP.load('../train/word1.txt.npy') # load
logreg = LogisticRegression()
logreg.fit(vectors, labels)
model = Word2Vec.load("../train/word2vec.model")

def summation(res,vec):
    new=[]
    l=0
    for i in res:
        new.append(i+vec[l])
        l+=1
    return new

with open('../train/dict.json') as f:
    words_vec = json.loads(f.read())

documents=[]

print("Testing on positive reviews")
for file_name in os.listdir('pos'):
    file_path = os.path.join('pos', file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        string=''
        for line in file:
            line=line.strip() # strip() removes all trailing and starting whitespaces as "   word  " becomes "word"                    
            for word in re.split(r"[^A-Za-z]+", line):          #split on any of the character except A-Z,a-z
                word = word.lower().strip()
                if len(word) == 0:
                    continue
                string=string+' '+word
 
        documents.append(string)
n,p=0,0
for doc in documents:
    words=doc.split()
    count=0
    res=[]
    for word in words:
        if word in model.wv and word in words_vec:
            count+=1
            if len(res)==0:
                for i in model.wv[word]:
                    res.append(i)
            else:
                res=summation(res,model.wv[word])
    l=0
    if len(res)==0:
        continue
    for l in range(0,len(res)):
        res[l]=(res[l]/count)
    predicted_sentiment = logreg.predict([res])
    if predicted_sentiment[0]=='pos':
        p+=1
    else:
        n+=1

print('pos:',p)
print('neg:',n)
    
docs=[]

print("Testing on Negative reviews")
for file_name in os.listdir('neg'):
    file_path = os.path.join('neg', file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        string=''
        for line in file:
            line=line.strip() # strip() removes all trailing and starting whitespaces as "   word  " becomes "word"                    
            for word in re.split(r"[^A-Za-z]+", line):          #split on any of the character except A-Z,a-z
                word = word.lower().strip()
                if len(word) == 0:
                    continue
                string=string+' '+word
 
        docs.append(string)
nn,np=0,0
for doc in docs:
    words=doc.split()
    count=0
    res=[]
    for word in words:
        if word in model.wv and word in words_vec:
            count+=1
            if len(res)==0:
                for i in model.wv[word]:
                    res.append(i)
            else:
                res=summation(res,model.wv[word])
    l=0
    if len(res)==0:
        continue
    for l in range(0,len(res)):
        res[l]=(res[l]/count)
    predicted_sentiment = logreg.predict([res])
    if predicted_sentiment[0]=='pos':
        np+=1
    else:
        nn+=1

print('neg:',nn)
print('pos:',np)

print("Evaluation")

print("Accurracy of model:",(nn+p)/25000)
print("Recall of Positive reviews:",(p)/(p+n))
print("Precision of Positive reviews:",(p)/(p+np))

x=[0.1]
y=[0.1]
x.append(((p)/(p+n)+(nn)/(np+nn))/2)   #recall
y.append(((p)/(p+np)+(nn)/(n+nn))/2)#precison
t=[0.5]
xpoints = NP.array(x)
ypoints = NP.array(y)

arr=[0.2,0.4,0.6,0.8]

for co in arr:
    nn,np=0,0
    for doc in docs:
        words=doc.split()
        count=0
        res=[]
        for word in words:
            if word in model.wv and word in words_vec:
                count+=1
                if len(res)==0:
                    for i in model.wv[word]:
                        res.append(i)
                else:
                    res=summation(res,model.wv[word])
        l=0
        if len(res)==0:
            continue
        for l in range(0,len(res)):
            res[l]=(res[l]/count)
        #predicted_sentiment = logreg.predict([res])
        predicted_probabilities = logreg.predict_proba([res])
        probability_positive_sentiment = predicted_probabilities[0, 1]
        if probability_positive_sentiment>=i:
            np+=1
        else:
            nn+=1

    n,p=0,0
    for doc in documents:
        words=doc.split()
        count=0
        res=[]
        for word in words:
            if word in model.wv and word in words_vec:
                count+=1
                if len(res)==0:
                    for i in model.wv[word]:
                        res.append(i)
                else:
                    res=summation(res,model.wv[word])
        l=0
        if len(res)==0:
            continue
        for l in range(0,len(res)):
            res[l]=(res[l]/count)
        predicted_probabilities = logreg.predict_proba([res])
        probability_positive_sentiment = predicted_probabilities[0, 1]
        if probability_positive_sentiment>=co:
            p+=1
        else:
            n+=1
    
    
    x.append(((p)/(p+n)+(nn)/(np+nn))/2)   #recall
    y.append(((p)/(p+np)+(nn)/(n+nn))/2)  #precison
    t.append(co)


xpoints = NP.array(x)
ypoints = NP.array(y)

plt.plot(xpoints, ypoints,marker='o')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

f=[]
all_labels=[]
i=0
while i < len(t):
    f.append((x[i],y[i],(2*x[i]*y[i])/(x[i]+y[i]),t[i]))
    i+=1

f.sort(key=lambda a: a[0],reverse=True)
print("Best threshold value for high Recall:",f[0][3])
print("Recall:",f[0][0])
print("Precision:",f[0][1])
print("F measure:",f[0][2])
f.sort(key=lambda a: a[1],reverse=True)
print("Best threshold value for high Precision",f[0][3])
print("Recall:",f[0][0])
print("Precision:",f[0][1])
print("F measure:",f[0][2])
f.sort(key=lambda a: a[2],reverse=True)
print("Best threshold value for high Precision and Recall:",f[0][3])
print("Recall:",f[0][0])
print("Precision:",f[0][1])
print("F measure:",f[0][2])


plt.show()


    

