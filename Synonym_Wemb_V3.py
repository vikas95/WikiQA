### This finds top 15 synonyms of each word in the vocabulory. This is an upgrade of Synonym_Wemb_V1
import heapq
import math
import ast
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))


"""
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
"""
#becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
# glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
f = open('glove.840B.300d.txt','r', encoding='utf-8')
# f = open('deps.contexts','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in f:
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
print("Embedding size is: ", emb_size)


def synonym(term1, WordEmb, threshold):

    max_dot_val=[]
    synonym_word=[]


    if term1 not in WordEmb.keys():  ###### Discuss this with Becky, we have to do something about this.
       synonym_word=term1

    else:

        for curr_key in WordEmb.keys():
            #if lmtzr.lemmatize(curr_key)!= term1:   ## we dont want synonyms or close words with same parent word, hence lemmatization
            if len(WordEmb[curr_key])==len(WordEmb[term1]):
                val=np.dot(WordEmb[term1],WordEmb[curr_key])
                if len(max_dot_val)<threshold:
                   max_dot_val.append(val)
                   synonym_word.append(curr_key)
                   if len(max_dot_val)==threshold:
                      max_dot_val=np.asarray(max_dot_val)
                      max_indices=np.argsort(-max_dot_val)
                      max_dot_val = -np.sort(-max_dot_val)
                      new_synonym_list = []
                      for ind1 in max_indices:
                          new_synonym_list.append(synonym_word[ind1])
                      synonym_word=new_synonym_list

                else:
                   if val>max_dot_val[-1]:
                      max_dot_val=np.append(max_dot_val,val)
                      synonym_word.append(curr_key)
                      new_synonym_list=[]
                      max_indices=np.argsort(-max_dot_val)
                      max_dot_val=-np.sort(-max_dot_val)
                      max_dot_val=max_dot_val[:-1]  ## don't take the last val
                      max_indices=max_indices[:-1]
                      for ind1 in max_indices:
                          new_synonym_list.append(synonym_word[ind1])

                      if len(new_synonym_list)!=threshold:
                         print("recheck, there is some error.  ")

                      synonym_word=new_synonym_list






    return synonym_word


def Question_synonym(ques1, WordEmb):
    New_ques1=""
    for term1 in ques1:
        syn_term1,oppterm1=synonym(term1,WordEmb)
        New_ques1=New_ques1+" "+str(term1)+" "+str(syn_term1)

    return New_ques1



#word_vectors = KeyedVectors.load_word2vec_format('ss_qz_04.dim50vecs.txt', binary=False)

#word_vectors.most_similar("blood")

#syn1, ant1 = synonym("cell",embeddings_index)

All_terms=[]


file = open('WikiQA-train.tsv',"r")

All_terms=[]
Candidate_answers=[]
Cand_ans=[]
Correct_ans=[]
Predicted_ans=[]
prev_ques_num=""
count=0
for line in file:
    if count==0:
       pass
    else:
        line=line.strip()
        columns=line.split("\t")
        ques_num=columns[0]
        if ques_num==prev_ques_num:  ## means we are at other rows of the same question.

           Cand_ans.append(columns[5])
           if columns[6]==1:
              Correct_ans.append(len(Cand_ans)-1)  ##### coz we start our candidate index from 0
        else:
           Candidate_answers.append(Cand_ans)
           Cand_ans=[]
           curr_ques=columns[1]

           curr_ques = tokenizer.tokenize(curr_ques.lower())
           curr_ques = [lmtzr.lemmatize(w1) for w1 in curr_ques]
           curr_ques = [w for w in curr_ques if not w in stop_words]

           All_terms+=curr_ques
           prev_ques_num=columns[0]

    count+=1



print (len(All_terms))
All_terms=list(set(All_terms))
print ("all unique terms",len(All_terms))

Synonym_terms={}
threshold=15
for ind1, term1 in enumerate(All_terms):
    if ind1%20==0:
       print(ind1)
    #print (term1)
    syn_term1=synonym(term1,embeddings_index,threshold)
    # print(term1," ",syn_term1)
    Synonym_terms.update({term1:syn_term1})

print (len(Synonym_terms))

syn_file=open("synonyms_ques_top_15.txt","w")
syn_file.write(str(Synonym_terms))

#New_ques_file=open("Synonym_added_questions.txt","w")
