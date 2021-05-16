#!/usr/bin/env python
# coding: utf-8

# In[28]:


'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
import re
import math


# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath,limitStr):
    # convert the limit argument from a string to an int
    limit = int(limitStr)
  
    # start lists for spam and ham email texts
    hamtexts = []
    spamtexts = []
    os.chdir(dirPath)
    # process all files in directory that end in .txt up to the limit
    #    assuming that the emails are sufficiently randomized
    for file in os.listdir("./spam"):
        if (file.endswith(".txt")) and (len(spamtexts) < limit):
            # open file for reading and read entire file into a string
            f = open("./spam/"+file, 'r', encoding="latin-1")
            spamtexts.append (f.read())
            f.close()
    for file in os.listdir("./ham"):
        if (file.endswith(".txt")) and (len(hamtexts) < limit):
            # open file for reading and read entire file into a string
            f = open("./ham/"+file, 'r', encoding="latin-1")
            hamtexts.append (f.read())
            f.close()
  
      # print number emails read
    print ("Number of spam files:",len(spamtexts))
    print ("Number of ham files:",len(hamtexts))

  
    # create list of mixed spam and ham email documents as (list of words, label)
    emaildocs = []
    # add all the spam
    for spam in spamtexts:
        tokens = nltk.word_tokenize(spam)
        emaildocs.append((tokens, 'spam'))
    # add all the regular emails
    for ham in hamtexts:
        tokens = nltk.word_tokenize(ham)
        emaildocs.append((tokens, 'ham'))
  
    # randomize the list
    random.seed(101)
    random.shuffle(emaildocs)
  
    # print a few token lists
    for email in emaildocs[:4]:
        print (email)
    
  #####################################################################################################################
  # possibly filter tokens
    def alpha_filter(w): # function that takes a word and returns true if it consists only of non-alphabetic characters
        pattern = re.compile('^[^a-z]+$')
        if (pattern.match(w)):
            return True
        else:
            return False    

    
    ###################################################################################################################
    # continue as usual to get all words and create word features
    
    # Creating a stopwords list to filter stopwords from email documents data
    nltkstopwords = nltk.corpus.stopwords.words('english')
    morestopwords =  ["Subject","subject","com","http",'could','would','might','must','need','sha','wo',"ll","t","m","re","ve","th","pm","e",'y',"s","e","d"]
    stopwords = nltkstopwords + morestopwords
  
    
    # Creating word features
    
    all_words_list = [(w,t) for (e,t) in emaildocs for w in e]
    words_alpha = [(w,t) for (w,t) in all_words_list if not alpha_filter(w)]
    words_stopped = [(w,t) for (w,t) in words_alpha if not w in stopwords]
    
    #frequency distribution of the words without filtering, for 2500 most common words
    FD_allwords = nltk.FreqDist(w for (w,t) in all_words_list)
    words_mc = FD_allwords.most_common(2500)
    print("\nFreqDist of 30 most common without filters")
    for w in words_mc[:30]:
        print(w)
        
    words_features = [w for (w,freq) in words_mc]
    
    # Creating frequency distribution of the words with filtering, for 2500 most common words
    FD_words_fil = nltk.FreqDist(word for (word,tag) in words_stopped)
    fil_words_mc = FD_words_fil.most_common(2500)
    print("\nFreqDist of 30 most common with filters")
    for word in fil_words_mc[:30]:
        print(word)
    
    fil_words_features = [word for (word,freq) in fil_words_mc]

    # feature sets from a feature definition function
    # define a feature definition function here
    def document_features(document, word_features):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['V_{}'.format(word)] = (word in document_words)
        return features
    
    # feature set with filtering
    featuresets = [(document_features(d,words_features), c) for (d,c) in emaildocs]
    
    # feature set with filtering
    fil_featuresets = [(document_features(d,fil_words_features), c) for (d,c) in emaildocs]
    
    
 ####################################################################################################################     
    #we’ll create some bigram features.
    #If we want to use highly frequent bigrams, we need to filter out special characters, which were very frequent in the bigrams, and also filter by frequency. 
    #The bigram pmi measure also required some filtering to get frequent and meaningful bigrams.

    #But there is another bigram association measure that is more often used to filter bigrams for classification features. 
    #This is the chi-squared measure, which is another measure of information gain, but which does its own frequency filtering. Another frequently used alternative is to just use frequency, which is the bigram measure raw_freq.
    #But there is another bigram association measure that is more often used to filter bigrams for classification features. This is the chi-squared measure, which is another measure of information gain, but which does its own frequency filtering. Another frequently used alternative is to just use frequency, which is the bigram measure raw_freq.
  
    # We’ll start by importing the collocations package and creating a short cut variable name for the bigram association measures.
    
    #We create a bigram collocation finder using the original movie review words, since the bigram finder must have the words in order. Note that our all_words_list has exactly this list.
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_words_list)
    
    #We use the chi-squared measure to get bigrams that are informative features. Note that we don’t need to get the scores of the bigrams, so we use the nbest function which just returns the highest scoring bigrams, using the number specified.
    bigram_features = finder.nbest(bigram_measures.chi_sq, 500)

    #The nbest function returns a list of significant bigrams in this corpus, and we can look at some of them.

    print(bigram_features[:50])
    
    #We are going to use these bigrams as features in a new features function.  
    #In order to test if any bigram in the bigram_features list is in the document, we need to generate the bigrams of the document, 
    #which we do using the nltk.bigrams function.
    
    #Now we create a feature extraction function that has all the word features as before, but also has bigram features.

    def bigram_document_features(document, word_features, bigram_features):
        document_words = set(document)
        document_bigrams = nltk.bigrams(document)
        features = {}
        for word in word_features:
            features['V_{}'.format(word)] = (word in document_words)
        for bigram in bigram_features:
            features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
        return features
    
    #Now we create feature sets as before, but using this feature extraction function.

    bigram_featuresets = [(bigram_document_features(d,words_features,bigram_features), c) for (d,c) in emaildocs]
    

###################################################################################################################################################

    #Representing Negation.
    
    negationwords = ['nowhere', 'nothing', 'noone','no', 'not', 'never', 'none','nobody',  
                     'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
    
    def Neg_features(document, word_features, negationwords):
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = False
            features['contains(NOT{})'.format(word)] = False
        # go through document words in order
        for ctr in range(0, len(document)):
            word = document[ctr]
            if ((ctr + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
                ctr += 1
                features['contains(NOT{})'.format(document[ctr])] = (document[ctr] in word_features)
            else:
                features['contains({})'.format(word)] = (word in word_features)
        return features

    Neg_featuresets = [(Neg_features(d, fil_words_features, negationwords), c) for (d, c) in emaildocs]


############# Representing LIWC SEntiment lexicon #######################################################################
    os.chdir("C:/Users/HP/Documents/SU - MSBA/MSBA Lectures/Spring 21/IST 664 NLP/Project/FinalProjectData/kagglemoviereviews/SentimentLexicons")
    def read_words():
        
        poslist = []
        neglist = []
        flexicon = open('liwcdic2007.dic', encoding='latin1')
        # read all LIWC words from file
        wordlines = [line.strip() for line in flexicon]
        # each line has a word or a stem followed by * and numbers of the word classes it is in
        # word class 126 is positive emotion and 127 is negative emotion
        for line in wordlines:
            if not line == '':
                items = line.split()
                word = items[0]
                classes = items[1:]
                for c in classes:
                    if c == '126':
                        poslist.append( word )
                    if c == '127':
                        neglist.append( word )
        return (poslist, neglist)

    # Getting positive and negative words
    (poslist, neglist) = read_words()

    # Creating new feature function
    def LIWC_features(document,poslist,neglist):
        doc_words = set(document)
        features = {}
        for word in poslist:
            features['P_{}'.format(word)] = (word in doc_words)
        for word in neglist:
            features['N_{}'.format(word)] = (word in doc_words)
        return features
    
    LIWC_featuresets = [(LIWC_features(d,poslist,neglist), c) for (d,c) in emaildocs]


        
####################################################################################################################################################################


    # B - Implementing additional features - TF_IDF score as value of word_features
    
    # IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    doc_w = [word for (email,tag) in emaildocs for word in email]
    dw = set(doc_w)
    FD_idf = nltk.FreqDist(dw)
    len_ed = len(emaildocs)    
    def IDF(word):
        if not FD_idf[word] == 0:
            IDF = math.log(len_ed/FD_idf[word])
            return IDF
        
    def TFIDF_features(document, word_features):
        dw = set(document)
        FD_dw = nltk.FreqDist(dw)
        features = {}
        for word in word_features:
            # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
            TF_word = FD_dw[word]/len(dw)
            idf_word = IDF(word)
            features['TFIDF_{}'.format(word)] = TF_word*idf_word
        return features
    
    TFIDF_featuresets = [(TFIDF_features(d,fil_words_features), c) for (d,c) in emaildocs]

        
##### train classifier and show performance in cross-validation#####################################################################
    
    #NLTK does not have a built-in function for cross-validation, but we can program the process in a function that takes the number of folds and the feature sets, 
    #and iterates over training and testing a classifier.  This function only reports accuracy for each fold and for the overall average.
    
    def eval_measures(gold,predicted):
        labels =list(set(gold))
        recall_list = []
        precision_list = []
        F1_list = []
        for lab in labels:
            TP=TN=FP=FN = 0
            for i,val in enumerate(gold):
                if val == lab and predicted[i] == lab:  TP += 1
                if val == lab and predicted[i] != lab:  FN += 1
                if val != lab and predicted[i] == lab:  FP += 1
                if val != lab and predicted[i] != lab:  TN += 1
            recall = TP/(TP+FP)
            precision = TP/(TP+FN)
            recall_list.append(recall)
            precision_list.append(precision)
            F1_list.append(2*(recall*precision)/(recall+precision))
        print("\tPrecision\tRecall\t\tF1")
        for i,lab in enumerate(labels):
            print(lab,"\t","{:10.3f}".format(precision_list[i]),"{:10.3f}".format(recall_list[i]),"{:10.3f}".format(F1_list[i]))
        
    def cross_validation_accuracy(num_folds, featuresets):
        subset_size = int(len(featuresets)/num_folds)
        print('Each fold size:', subset_size)
        accuracy_list = []
        # iterate over the folds
        for i in range(num_folds):
            test_this_round = featuresets[(i*subset_size):][:subset_size]
            train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
            # train using train_this_round
            classifier = nltk.NaiveBayesClassifier.train(train_this_round)
            # evaluate against test_this_round and save accuracy
            accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
            print (i, accuracy_this_round)
            accuracy_list.append(accuracy_this_round)
            goldlist = []
            predictedlist = []
            for (features, label) in test_this_round:
                goldlist.append(label)
                predictedlist.append(classifier.classify(features))
            eval_measures(goldlist,predictedlist)
        # find mean accuracy over all rounds
        print ('mean accuracy', sum(accuracy_list) / num_folds)
        
    # Classification and cross validation for emailwords WITHOUT Filtering
    train_set, test_set = featuresets[1000:], featuresets[:1000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier.show_most_informative_features(30)
    print("\nCross-validation reults for featuresets without filtering")
    cross_validation_accuracy(5,featuresets)
    
    # Classification and cross validation for emailwords WITH Filtering
    fil_train_set, fil_test_set = fil_featuresets[1000:], fil_featuresets[:1000]
    fil_classifier = nltk.NaiveBayesClassifier.train(fil_train_set)
    fil_classifier.show_most_informative_features(30)
    print("\nCross-validation reults for featuresets with filtering")
    cross_validation_accuracy(5,fil_featuresets)
    
    ## Classification and cross validation for word features and bigram features
    bigram_train_set, bigram_test_set = bigram_featuresets[1000:], bigram_featuresets[:1000]
    bigram_classifier = nltk.NaiveBayesClassifier.train(bigram_train_set)
    bigram_classifier.show_most_informative_features(30)
    print("\nCross-validation reults for bigram featuresets")
    cross_validation_accuracy(5,bigram_featuresets)
    
    # Negation Classification and cross validation for emailwords
    Neg_train_set, Neg_test_set = Neg_featuresets[1000:], Neg_featuresets[:1000]
    Neg_classifier = nltk.NaiveBayesClassifier.train(Neg_train_set)
    print("\nNegation Classifier accuracy")
    cross_validation_accuracy(5,Neg_featuresets)
    
    # LIWC Sentiment Lexicon Classification and cross-validation for emailwords
    LIWC_train_set, LIWC_test_set = LIWC_featuresets[1000:], LIWC_featuresets[:1000]
    LIWC_classifier = nltk.NaiveBayesClassifier.train(LIWC_train_set)
    print("\nLIWC Sentiment Classifier accuracy")
    cross_validation_accuracy(5,LIWC_featuresets)   


    # TFIDF Classification and cross-validation for emailwords
    TFIDF_train_set, TFIDF_test_set = TFIDF_featuresets[1000:], TFIDF_featuresets[:1000]
    TFIDF_classifier = nltk.NaiveBayesClassifier.train(TFIDF_train_set)
    print("\TFIDF Classification accuracy")
    cross_validation_accuracy(5,TFIDF_featuresets)   

    
    
    
############# Representing classification using Sci Kit Learn classifier with features produced in NLTK #######################################################################


    # B - Using Sci-kit learn 
    def writeFeatureSets(featuresets, outpath):
        # open outpath for writing
        f = open(outpath, 'w')
        # get the feature names from the feature dictionary in the first featureset
        featurenames = featuresets[0][0].keys()
        # create the first line of the file as comma separated feature names
        #    with the word class as the last feature name
        featurenameline = ''
        for featurename in featurenames:
            # replace forbidden characters with text abbreviations
            featurename = featurename.replace(',','CM')
            featurename = featurename.replace("'","DQ")
            featurename = featurename.replace('"','QU')
            featurenameline += featurename + ','
        featurenameline += 'class'
        # write this as the first line in the csv file
        f.write(featurenameline)
        f.write('\n')
        # convert each feature set to a line in the file with comma separated feature values,
        # each feature value is converted to a string 
        #   for booleans this is the words true and false
        #   for numbers, this is the string with the number
        for featureset in featuresets:
            featureline = ''
            for key in featurenames:
                featureline += str(featureset[0][key]) + ','
            featureline += featureset[1]
            # write each feature set values to the file
            f.write(featureline)
            f.write('\n')
        f.close()

    # Using the filtered feature sets
    outpath = "SK_Learn_features.csv"
    writeFeatureSets(fil_featuresets, outpath)


    

# """
# commandline interface takes a directory name with ham and spam subdirectories
#    and a limit to the number of emails read each of ham and spam
# It then processes the files and trains a spam detection classifier.

# """
# if __name__ == '__main__':
#     if (len(sys.argv) != 3):
#         print ('usage: python classifySPAM.py <corpus-dir> <limit>')
#         sys.exit(0)
#     processspamham(sys.argv[1], sys.argv[2])
        


# In[29]:


processspamham("C:/Users/HP/Documents/SU - MSBA/MSBA Lectures/Spring 21/IST 664 NLP/Project/FinalProjectData/EmailSpamCorpora/corpus",1500)

