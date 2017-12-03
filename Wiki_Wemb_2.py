import heapq
import math
import ast
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))
# stop_words=[]
print(stop_words)
print(len(stop_words))


"""
Vocab_Wemb={}
Voc_file=open("Vocab_Wemb.txt","r")
for Voc_line in Voc_file:
    Voc_line=Voc_line.strip()
    Words=Voc_line.split()
    Emb=ast.literal_eval[Words[1:]]

    Vocab_Wemb.update({Words[0]:Emb})

print ("len of Vocab embeddings is: ", len(Vocab_Wemb))
"""

#becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
# f = open('glove.6B.100d.txt','r', encoding='utf-8')
f = open('glove.840B.300d.txt','r', encoding='utf-8')
# f = open("GW_vectors.txt", 'r', encoding='utf-8')  ## gives a lot lesser performance.

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



IDF_file=open("IDF_file_dev_alone_correct.txt","r")
for line in IDF_file:
    IDF=ast.literal_eval(line)
    break

# print ("questions length is: ", len(Question_set))
# print ("Candidate answers set length is: ", len(Candidate_answers[2]))
print ("len of IDF is: ",len(IDF))



def Ques_Emb(ques1, IDF, embeddings_index):
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    for q_term in ques1:
        if q_term in embeddings_index.keys():
           Ques_Matrix = np.append(Ques_Matrix, np.array([embeddings_index[q_term]]), axis=0)
           IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)

    return Ques_Matrix, IDF_Mat



def Word2Vec_score(curr_ques, Cand_ans, IDF, Word_Embs):
    max_score=0
    min_score=0
    Cand_ans_score=[]

    threshold_vals = len(curr_ques)  ## math.ceil     math.ceil(0.75 * float()
    # print("threshold value is: ",threshold_vals)
    Ques_Matrix, Ques_IDF = Ques_Emb(curr_ques, IDF, Word_Embs)
    cand_len=[]
    for cand1 in Cand_ans:
        cand_len.append(len(cand1))
    threshold=min(cand_len)


    for cand_a1 in Cand_ans:
        Cand_Matrix, Cand_IDF = Ques_Emb(cand_a1, IDF, Word_Embs)
        Cand_Matrix = Cand_Matrix.transpose()
        # print(Cand_Matrix.shape[1])
        if Cand_Matrix.size == 0 or Ques_Matrix.size == 0:
            print ("we have weird cases like these....................")
        else:

            Score = np.matmul(Ques_Matrix, Cand_Matrix)
            Score = np.sort(Score,axis=1)

            max_indices = np.argmax(Score, axis=1)
            min_indices = np.argmin(Score, axis=1)


            max_score1 = Score [:,-3:]   ## taking 3 highest element columns
            # print(max_score1)

            max_score1 = np.matmul(np.transpose(Ques_IDF), max_score1)
            max_score1 = np.asarray(max_score1).flatten()
            max_score1 = (sum(max_score1))
            max_score=max_score1

            min_score = Score[:, 0:2]
            min_score = np.matmul(np.transpose(Ques_IDF), min_score)
            min_score = np.asarray(min_score).flatten()
            # min_score = heapq.nsmallest(threshold_vals, min_score)  ## threshold=2
            min_score = (sum(min_score))


            total_score = max_score # - (min_score)
            # total_score = total_score # / float(Cand_Matrix.shape[1])
            Cand_ans_score.append(total_score)
    # print("Can scores are: ", Cand_ans_score)

    if len(Cand_ans_score)< 1: ## empty predicted_val which is not good, which basically means that we could not find any embeddings for the words. IDF shouldnt be a prob.
       predicted_val="empty"
    else:
       Cand_ans_score = np.asarray(Cand_ans_score)
       predicted_val=np.argmax(Cand_ans_score)

    return predicted_val




file = open('WikiQA-dev.tsv',"r")

Question_set={}
Candidate_answers={}
Cand_ans=[]
Correct_ans={}
Predicted_ans=[]
prev_ques_num=""
count=0
Question_numbers=[]
for line in file:
    if count==0:
       pass
    else:

        line=line.strip()
        columns=line.split("\t")
        ques_num=columns[0]
        Question_numbers.append(ques_num)

        if ques_num==prev_ques_num:  ## means we are at other rows of the same question.

           Cand_ans.append(columns[5])
           #if int(columns[6])==1:
              #Correct_ans.append(len(Cand_ans)-1)  ##### coz we start our candidate index from 0
        else:
           Candidate_answers.update({prev_ques_num:Cand_ans})
           Cand_ans=[]
           Question_set.update({columns[0]:columns[1]})
           Cand_ans.append(columns[5])
           prev_ques_num=columns[0]
           #if int(columns[6])==1:
              #Correct_ans.append(len(Cand_ans)-1)  ##### coz we start our candidate index from 0

        prev_ques_num = columns[0]
        if int(columns[6]) == 1:
            if columns[0] in Correct_ans.keys():
               Correct_ans[columns[0]].append((len(Cand_ans) - 1))
            else:
               Correct_ans.update({columns[0]:[(len(Cand_ans) - 1)]})  ##### coz we start our candidate index from 0

    count+=1

print ("Question len is: ",len(Question_set))
print ("Correct ans len is ", len(Correct_ans))
print ("Candidate ans set len is ", len(Candidate_answers))





accuracy=0
count=0
for valid_ques in Correct_ans.keys():
    if count%20==0:
       print (count)
    curr_ques=Question_set[valid_ques]
    curr_ques = tokenizer.tokenize(curr_ques.lower())
    curr_ques = [lmtzr.lemmatize(w1) for w1 in curr_ques]
    curr_ques = [w for w in curr_ques if not w in stop_words]

    cand_ans=Candidate_answers[valid_ques]
    for ins1, ans1 in enumerate(cand_ans):
        ans1=tokenizer.tokenize(ans1.lower())
        ans1 = [lmtzr.lemmatize(w1) for w1 in ans1]
        ans1 = [w for w in ans1 if not w in stop_words]
        cand_ans[ins1]=ans1


    Predictions = Word2Vec_score(curr_ques, cand_ans, IDF, embeddings_index)  # , Positive_term_threshold, Negative_term_threshold)
    # Predicted_ans.append(Predictions)
    unsolved_ques=0
    if Predictions=="empty":
       unsolved_ques+=1
    else:
       if Predictions in Correct_ans[valid_ques]:
          accuracy+=1
    count+=1

print ("accuracy for len is:  ",accuracy/float(len(Correct_ans)))

print ("number of unsolved questions are: ", unsolved_ques)
print ("this is for 3 pos terms NO neg terms ")
