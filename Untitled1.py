#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers


# In[2]:


pip install -U sentence-transformers


# In[3]:


import pandas as pd
file_name= r"testing (2).xlsx"
file =   r"training (2).xlsx"
dtest = pd.read_excel(file_name)
dtrain = pd.read_excel(file_name)


# In[4]:


model=SentenceTransformer('sentence-transformers/sentence-t5-base')
#dtrain['EmbeddingsLM']=dtrain['input'].apply(lambda x:model.encode(x))
dtest['EmbeddingsLM']=dtest['Equation'].apply(lambda x:model.encode(x))
#t5_train=pd.DataFrame(dtrain['EmbeddingsLM'].tolist(),index=data.index).add_prefix('embed_')
t5_test=pd.DataFrame(dtest['EmbeddingsLM'].tolist(),index=dtest.index).add_prefix('embed_')


# In[ ]:




