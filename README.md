# chat-bot-for-website
Chatbot is trained with the set of input questions and corresponding answers. So, whenever user enters a query bot responds to user with appropriate answer. In addition to this sentiment analysis is added to the chat to know the motive of the user. Language: Python. Packages used: pandas, chatterbot, BeautifulSoup, nltk, re, sklearn, nltk-corpus.


The motive behind this project is, many of the organizations/websites have to answers various queries to their users. Every time someone has to be there to answer their queries. So, using this chatbot it serves the following purpose. 1. Standardization of information being provided to users 2. Checking for FAQ’s 3. Learning from FAQ’s 4. Handling multiple users at a time

Format of Data Set: Data should be provided as following Question 1 Answer 1 Question 2 Answer 2 Question 3 Answer 3 ....... and so on. in CSV file everything in single column.(column name optional -here I gave it as 'QA')

Requirements: 1. Chatterbot, chatbot package 2.NLTK

Dis-Advantages: -> This doesn't understands the meaning in the text. It just finds the keywords in the question and matches appropriate answer.

Future Extension: The main motive of any chat-bot is to understand what user requirement is and respond to him accordingly. This feature of understanding the meaning in the question is missing above code. So we have to move to deep learning techniques in-order understand the meaning in the text. There are several algorithms available. What to choose is based on user interest.

Google released open source TensorFlow ( https://www.tensorflow.org/) for deep learning techniques.

Seq2seq Algorithm: Here I'm going to use seq2seq algorithm for training my chat-bot. So what is seq2seq Algorithm ?? The model is based on two LSTM layers. One for encoding the input sentence into a "thought vector", and another for decoding that vector into a response. This model is called Sequence-to-sequence or seq2seq.
