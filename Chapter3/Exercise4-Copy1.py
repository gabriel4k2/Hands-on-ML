#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import tarfile
import html2text
import string
import re
from nltk.stem.porter import *

from email.parser import Parser
from email.policy import default
from sklearn.base import *
from sklearn.datasets import load_files
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer


dataset_path = "dataset/"
url_pattern = r'https?://\S+'
number_pattern= r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?'
only_easy_ham = True
email_parser = None
htmlconverter = None
stemmer = None

def extract_bz2(filename, path="."):
    with tarfile.open(filename, "r:bz2") as tar:
        tar.extractall(path)

def removeHardHam(data, targets, hard_ham_index):
    _to_delete = []
    for _index in range(len(data)):
        if targets[_index] == int(hard_ham_index):
            _to_delete.append(_index)
        
    return np.delete(data, _to_delete),  np.delete(targets, _to_delete)


#The dataset has multiple labels such as easy_ham, easy_ham2,
# we want to simply convert to ham or spam
def binarizeLabels(targets):
    for _index in range(targets.size):
        if "ham" in targets[_index]:
            targets[_index] = "ham"
        else:
            targets[_index] = "spam"
            
    return targets

"""
processMessageObj 
Helper function only process text/plain or text/html. Images/gifs 
or other multimedia are not useful. Multiparts are also not
useful because they will be visited later with walk()

"""
def processMessageObj(message_object):
    _type = message_object.get_content_type()
    _body = None
    if _type == "text/plain":
        _body = message_object.get_payload(decode=True)
        _body = _body.decode('latin-1')
    elif _type == "text/html":
        #TODO preprocess html
        _body = message_object.get_payload(decode=True)
        _body = _body.decode('latin-1')
        _body = transformHTMLtoPlain(_body)

    return _body

def transformHTMLtoPlain(_str, ignore_images=True):
    global htmlconverter


    if htmlconverter is None:
        htmlconverter = html2text.HTML2Text()

    if ignore_images:
        htmlconverter.ignore_images = True

    inspect_result = htmlconverter.handle(_str)

    return inspect_result


def stemmfy(message):
    global stemmer
    _new_body = None
    if  stemmer is None:
        stemmer = PorterStemmer()

    for word in message.split():
        temp = stemmer.stem(word)
        if _new_body is None:
            _new_body = temp
        else:
            _new_body = _new_body + " " + temp

    return _new_body


def getEmailSubject(email_message ):
    global email_parser

    if email_parser is None:
        email_parser = Parser(policy=default)

    email_message = email_parser.parsestr(email_message.decode('iso-8859-1')) #UTF-8 does not work (some email have latin chars

    return email_message["subject"]

"""
Processes the email "payload", that is, the actual message. This processing includes parsing the byte stream
into an email object, thenn decoding it and finally walking through the "sections"

input: email_message = a numpy array element, representing a byte stream

"""
def processEmailPayload(email_message, to_lower_case, stemm=True, remove_punct = True, substitute_number=True):
    body = ""
    global email_parser

    if email_parser is None:
        email_parser = Parser(policy=default)

    email_message = email_parser.parsestr(email_message.decode('iso-8859-1'))
    #UTF-8 does not work (some email have latin chars)



    if email_message.is_multipart():
        for part in email_message.walk():
            _temp = processMessageObj(part)
            if _temp is not None:
                body = body + _temp

    else:
        _temp  =  processMessageObj(email_message)
        body = body + _temp if _temp is not None else body


    if to_lower_case:
        body = body.lower()

    #URLS are changed always (hardcoded) it makes no sense to allow them (too much noise)
    body = re.sub(pattern=url_pattern, repl=' _URL_ ', string=body)

    
    if substitute_number:
        body = re.sub(pattern=number_pattern, repl='NUMBER', string=body)
        
    if remove_punct:
        body = body.translate((str.maketrans('', '', string.punctuation)))



        
    if stemm:
        body = stemmfy(body)
    #If the email message is a html message for instance, the body will be reduced to "none", then simply convert it
    #to a null string (to not crash further steps)...
    return body if body is not None else ""



def convertIndexToString(targets, index_to_string_dict):
    new_targets = []
    for _target in targets:
        new_targets.append( index_to_string_dict.get(_target) )
    
    return np.asarray(new_targets )

class PreprocessStrToEmail(BaseEstimator, TransformerMixin):
    def __init__(self,   to_lower_case= False, stemm=True, remove_punct=True,                  substitute_number = True): # no *args or **kargs
        
        self.to_lower_case = to_lower_case
        self.stemm = stemm
        self.remove_punct = remove_punct
        self.substitute_number =  substitute_number

    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        _parser = Parser(policy=default)

        return np.array([processEmailPayload(xi, self.to_lower_case, self.stemm, self.remove_punct                                             ,   self.substitute_number ) for xi in X])
 




        


# In[3]:


os.chdir("SpamAssassin")
compressed_files = [x for x in os.listdir()  if x.endswith(".bz2")]
for _file in compressed_files:    
    extract_bz2(_file, path=dataset_path)


# In[15]:


_dataset = load_files(dataset_path)
_data = np.array(_dataset.data)
_target = _dataset.target
_target_names =  _dataset.target_names


# As we can see below, every data has a categorical label contained in the following set: 

# In[16]:


print(_target_names)


# In[17]:


if only_easy_ham:
    _hard_ham_index = _target_names.index('hard_ham')
    _data, _target = removeHardHam(_data, _target, _hard_ham_index)


# It's desirable to simply set the labels as whether they are ham/spam:

# In[18]:


index_to_str_label = dict()
zipped = zip(list(range(_target_names.__len__())), _target_names )
for x,y in zipped:
    index_to_str_label.update({x:y})

_target = convertIndexToString(_target,index_to_str_label )
_target = binarizeLabels(_target)

print(_target[:16])


# As we can see below there is a lot of noise in the message, such as the protocol used which IP sent the message, etc...

# In[19]:


print(_data[1])


# The preprocessing will take care to only get the message proper and the subject if the user wants. The other preprocessing hyperparameters are whether to convert to_lower_case, stemm the words and remove punctuation. Below is a fully processed email message.

# In[23]:


pre_process = PreprocessStrToEmail(to_lower_case=True, stemm=True, remove_punct = True )
_data_processed_example = pre_process.fit_transform([_data[2]])


# In[24]:


print(_data_processed_example)


# Below is sightly less preprocessed message. As we can see there is significantly more noise.

# In[25]:


pre_process_weaker = PreprocessStrToEmail(to_lower_case=True, stemm=False, remove_punct = True, substitute_number=False)
_data_processed_example = pre_process_weaker.fit_transform([_data[2]])


# In[26]:


print(_data_processed_example)


# In[28]:


_data_processed = pre_process.fit_transform(_data[0:1000])


# In[ ]:





# In[29]:


count_vect = CountVectorizer()


# In[34]:


X_train_counts = count_vect.fit_transform(_data_processed)

print("gah")


# In[37]:




# In[ ]:




