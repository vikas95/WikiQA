import math
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')

import ast

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#lmtzr.lemmatize('cars')
import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))


###################

file = open('WikiQA-dev.tsv',"r")

Question_set={}
Candidate_answers={}
Cand_ans=[]
Correct_ans={}
Predicted_ans=[]
prev_ques_num=""
count=0
Question_numbers=[]
IDF={}

Total_doc=0
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


           curr_ques=columns[1]
           curr_ques = tokenizer.tokenize(curr_ques.lower())
           curr_ques = [lmtzr.lemmatize(w1) for w1 in curr_ques]
           curr_ques = [w for w in curr_ques if not w in stop_words]
           curr_ques_set = list(set(curr_ques))
           for qt1 in curr_ques_set:
               if qt1 in IDF.keys():
                   IDF[qt1] += 1
               else:
                   IDF.update({qt1: 1})

           Total_doc+=1 ## 1 for the question as the document

           for answers in Candidate_answers[prev_ques_num]:
               Total_doc+=1  ## 1 for each cand ans
               curr_ans=tokenizer.tokenize(answers.lower())
               curr_ans = [lmtzr.lemmatize(w1) for w1 in curr_ans]
               curr_ans = [w for w in curr_ans if not w in stop_words]
               curr_ans_set = list(set(curr_ans))
               for qt1 in curr_ans_set:
                   if qt1 in IDF.keys():
                       IDF[qt1] += 1
                   else:
                       IDF.update({qt1: 1})



           prev_ques_num = columns[0]
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


for each_word in IDF.keys():
    doc_count=IDF[str(each_word)]

    IDF[str(each_word)]=math.log10((Total_doc-doc_count+0.5)/float(doc_count+0.5))


file=open("IDF_file_dev_alone_correct.txt","w")
file.write(str(IDF)) ####### wronggggggggggggg




"""
accuracy=0
for valid_ques in Correct_ans.keys():
    cand_ans=Candidate_answers[valid_ques]
    pred1=Len_algo(valid_ques,cand_ans)
    if pred1 in Correct_ans[valid_ques]:
       accuracy+=1

"""

