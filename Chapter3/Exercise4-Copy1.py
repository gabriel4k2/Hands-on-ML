#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os, tarfile
import email
import numpy as np  
from sklearn.datasets import load_files
from sklearn.base import *
from sklearn.model_selection import StratifiedShuffleSplit
from email.parser import BytesParser, Parser
from email.policy import default
dataset_path = "dataset/"
only_easy_ham = True
email_parser = None

def extract_bz2(filename, path="."):
    with tarfile.open(filename, "r:bz2") as tar:
        tar.extractall(path)

def removeHardHam(data, targets, hard_ham_index):
    _to_delete = []
    for _index in range(len(data)):
        if targets[_index] == int(hard_ham_index):
            _to_delete.append(_index)
        
    return np.delete(data, _to_delete),  np.delete(targets, _to_delete)

def get_data_index_from_name(name):
    pass

#The dataset has multiple labels such as easy_ham, easy_ham2,
# we want to simply convert to ham or spam
def binarizeLabels(targets):
    for _index in range(targets.size):
        if "ham" in targets[_index]:
            targets[_index] = "ham"
        else:
            targets[_index] = "spam"
            
    return targets

'''
getEmailPayload helper
Only process text/plain or text/html. Images/gifs or other multimedia are not useful. Multiparts are also not
useful because they will be visited later with walk()

'''
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
    return _body

def getEmailPayload(email_message):
    body = ""
    global email_parser

    if email_parser is None:
        email_parser = Parser(policy=default)

    email_message = email_parser.parsestr(email_message.decode('iso-8859-1')) #UTF-8 does not work (some email have latin chars

    if email_message.is_multipart():
        for part in email_message.walk():
            _temp = processMessageObj(part)
            if _temp is not None:
                body = body + _temp
            # if part.is_multipart():
            #     for subpart  in part.get_payload(): #when multipart, a list is returned
            #         body = body + subpart.get_payload()

            # else:
            #     if part.get_all('Content-Transfer-Encoding') == 'base64':
            #         pass #converter base64 para string
    else:
        body = processMessageObj(email_message)
    return body



def convertIndexToString(targets, index_to_string_dict):
    new_targets = []
    for _target in targets:
        new_targets.append( index_to_string_dict.get(_target) )
    
    return np.asarray(new_targets )

class PreprocessStrToEmail(BaseEstimator, TransformerMixin):
    def __init__(self, only_body = True): # no *args or **kargs
        self.only_body = only_body
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        _parser = Parser(policy=default)
        #parsestr(_data[0].decode('UTF-8'))
        #_X = np.asarray(X)
        #_X = np.array([_parser.parsestr(x.decode('UTF-8'))] for x in _X)
        if self.only_body:
            #X = np.array(getEmailPayload(x) for x in X)
           # X = getEmailPayload(X)
            X = np.array(list(map(getEmailPayload, X)))
            #temp = np.apply_along_axis(func1d=getEmailPayload, axis=0, arr=X)

        else:
            pass
        return X


        


# In[2]:


os.chdir("SpamAssassin")
compressed_files = [x for x in os.listdir()  if x.endswith(".bz2")]
for _file in compressed_files:    
    extract_bz2(_file, path=dataset_path)


# In[3]:


_dataset = load_files(dataset_path)
_data = np.array(_dataset.data)
_target = _dataset.target
_target_names =  _dataset.target_names


# As we can see below, every data has a categorical label contained in the following set: 

# In[4]:


print(_target_names)


# In[5]:



if only_easy_ham:
    _hard_ham_index = _target_names.index('hard_ham')
    _data, _target = removeHardHam(_data, _target, _hard_ham_index)


# It's desirable to simply set the labels as whether they are ham/spam:

# In[6]:


index_to_str_label = dict()
zipped = zip(list(range(_target_names.__len__())), _target_names )
for x,y in zipped:
    index_to_str_label.update({x:y})

_target = convertIndexToString(_target,index_to_str_label )
_target = binarizeLabels(_target)

print(_target[:16])


# In[7]:


print(_data[0])


# In[8]:


headers = Parser(policy=default).parsestr(_data[0].decode('UTF-8'))


# In[9]:


print('To: {}'.format(headers['to']))


# In[10]:


print('From: {}'.format(headers['from']))


# In[11]:


elem = headers.get_payload()
type(elem)
print(elem[-1])


# In[12]:


#Let's see the ratio of spam to ham:
spam_occurrences = np.count_nonzero(_target == 'spam')
ham_occurrences = np.count_nonzero(_target == 'ham')
print(spam_occurrences/ham_occurrences)


# In[15]:


pre_process = PreprocessStrToEmail()
pre_process.fit_transform(_data)
_duh = getEmailPayload(_data[0])


# In[8]:


#Using stratifies shuffle to obtain a roughly similar ratio:
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(_data, _target):
    X_strat_train_set = _data[train_index]
    X_strat_test_set =  _data[test_index]
    Y_strat_train_set = _target[train_index]
    Y_strat_test_set = _target[test_index]


# In[10]:


#OK, the train set does accurately reflects the data
spam_occurrences = np.count_nonzero(Y_strat_train_set == 'spam')
ham_occurrences = np.count_nonzero(Y_strat_train_set == 'ham')
print(spam_occurrences/ham_occurrences)


# In[ ]:





# In[ ]:



