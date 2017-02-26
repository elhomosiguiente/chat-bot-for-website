#Import libraries
import pandas as pd
import os


###
#import data sets
Data1= pd.read_csv(r"..path1.csv")
Data2=pd.read_csv(r"..path2.csv")
Data3=pd.read_csv(r"..path3.csv")
##
#Size of each data set
print(Data1.shape)
print(Data2.shape)
print(Data3.shape)

####

#Pre processing the data
import re
def cleaning_text(review):
    cleaned =re.sub("[^a-zA-Z/:.?]", " ",review)
    words =cleaned.lower().split()
    return(" ".join( words ))
new_words=[]
for i in range(0,Data1.size -1):
    new_words.append(cleaning_text(Data1["QA"][i]))
for i in range(0,Data2.size -1):
    new_words.append(cleaning_text(Data2["QA"][i]))
for i in range(0,Data3.size -1):
    new_words.append(cleaning_text(Data3["QA"][i]))
    
 ####

from nltk import corpus
from chatterbot import ChatBot
import logging


# Uncomment the following line to enable verbose logging
# logging.basicConfig(level=logging.INFO)

# Create a new instance of a ChatBot
chatbot1 = ChatBot("Dummy1", 
    storage_adapter="chatterbot.storage.JsonFileStorageAdapter",
    logic_adapters=[
        "chatterbot.logic.MathematicalEvaluation",
        "chatterbot.logic.TimeLogicAdapter",
        "chatterbot.logic.BestMatch"
    ],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
    database="../database2.db"
)


#Import List trainer for training the chat history
#
from chatterbot.trainers import ListTrainer
#
#Move the preprocessed chat history to conversations
#
conversation=new_words
chatbot1.set_trainer(ListTrainer)
#
#training the conversation
#
chatbot1.train(conversation)
#
#Import corpus.english for basic greetings and casual replies
#
#chatbot1.train("chatterbot.corpus.english")
#chatbot1.set_trainer(ChatterBotCorpusTrainer)
#chatbot1.train("chatterbot.corpus.english")
#chatbot1.train(
#    "chatterbot.corpus.english.greetings",
#    "chatterbot.corpus.english.conversations"
#)

######

#_____Sentiment Analysis ________#


import pandas as pd
df1 = pd.read_csv(r"E:\Aegis\Project\Data_set\rechatterbotpdf/chat_sentiment.csv")

from bs4 import BeautifulSoup  
import nltk
import re
from nltk.corpus import stopwords


##
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 

print df1.columns

clean_review = review_to_words( df1["Questions"][0] )
print clean_review


num_reviews = df1["Questions"].size
clean_train_reviews = []

for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( df1["Questions"][i] ) )
    
    
from sklearn.feature_extraction.text import CountVectorizer

 
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print vocab

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, df1["Sentiment"] )



########
clean_test_reviews = [] 
clean_review = review_to_words("what is eligibility criteria" )
clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
print result



#######

print("Type something to begin...")

# The following loop will execute each time the user enters input
while True:
    try:
        # We pass None to this method because the parameter
        # is not used by the TerminalAdapter
        bot_input = chatbot1.get_response(None)
        
        clean_test_reviews = [] 
        clean_review = review_to_words(bot_input )
        clean_test_reviews.append( clean_review )

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()

        # Use the random forest to make sentiment label predictions
        result = forest.predict(test_data_features)
        print result

    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break
