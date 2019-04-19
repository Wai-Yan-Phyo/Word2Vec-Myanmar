
# coding: utf-8

# # Ko Ye Kyaw Thu Data 
# #Data 60000

# Three reliable model you can see...
# 1. Build Word2Vec with Skip Gram include negative value 3
# 3. Build Word2Vec Model with Skip Gram (Neg=0,Iter=2,size=100,window=5)
# 2. Build Word2Vec Model with Skip Gram (Neg=0,Iter=2,size=100,window=10)
# 

# # --------------------------******************************--------------------------------------

# # Hierarchical softmax

# #Here..... 
#          Hierarchical softmax uses a binary tree to represent all words in the vocabulary. The words themselves are leaves in the tree. For each leaf, there exists a unique path from the root to the leaf, and this path is used to estimate the probability of the word represented by the leaf. “We define this probability as the probability of a random walk starting from the root ending at the leaf in question.”

# https://adriancolyer.files.wordpress.com/2016/04/word2vec-hierarchical-softmax.png?w=600

# # ---------------------------------------*****------------------------------------------------

# # Why Negative sampling?????

# Negative Sampling is simply the idea that we only update a sample of output words per iteration. The target output word should be kept in the sample and gets updated, and we add to this a few (non-target) words as negative samples. “A probabilistic distribution is needed for the sampling process, and it can be arbitrarily chosen… One can determine a good distribution empirically.”

# https://adriancolyer.files.wordpress.com/2016/04/word2vec-subsampling.png?w=600

# f(wi) is the frequency of word wi and t is a chosen threshold, typically around 10-5.

# # ---------------------------------------------************---------------------------------

# # All Model Report 

# [model -----SkipGram with neg 3][model1 ------CBOW with neg 3][model2 ------ SkipGram,iter 2,window 5]||[model3 ----- SkipGram,iter 2,window 10]||[model4 ----- SkipGram,iter 2,window 15]||[model_w3 -----SkipGram,iter 2,window 3]

# https://docs.google.com/spreadsheets/d/1D9tVg5EVLsboBtV7Z9wbcjHpKvgjnwNzr1QvctBLDm0/edit#gid=0

# In[1]:


import pandas as pd
import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import FastText


# In[2]:


import logging
import datalab.storage as dlb_storage
from io import BytesIO


# Function For read_csv from GCS(cloud)

# In[3]:


def read_csv(file, nrows,usecols=all):
    stream =  dlb_storage.Bucket('project-bagandata').item(file).read_from()
    data = pd.read_csv(BytesIO(stream), engine='c', nrows=nrows, low_memory=False, header='infer', usecols=usecols)
    return data


# In[4]:


data=read_csv('nlp/KoYeKyawThu-data/mmlines.txt',60000,usecols=all)
print(type(data))
data.columns = ['sent']


# In[5]:


data.columns


# In[5]:


display (data.head())


# Convert DataFrame To List

# In[5]:


data = data.iloc[:,0].values.tolist()
print (data[:5])


# In[6]:


new_data = [s.split() for s in data]
print (new_data[:5])


# # Do not run this cell if you don't need to check the whole data

# In[7]:


##Careful start this cell only for check whole data
display (new_data)


# Set stop list 

# In[8]:


stoplist = set('သည် ။ ကို တွင် ပါ သာ ဟာ လာ မ သော ရှိ များ ၊ သည့် ၏ နှင့် ပြီး လည်း ၍ ဖြစ် သောအခါ ဖို့ သလဲ မှာ ကြ တို့ ခြင်း က ခဲ့'.split())


# In[9]:


texts = [[word for word in sentence if word not in stoplist]for sentence in new_data]


# In[10]:


texts[:5]


# # Build Word2Vec with Skip Gram include negative value 3
# Good

# In[11]:


get_ipython().run_cell_magic('time', '', 'model = Word2Vec(texts, size=100, window=10, min_count=5,sample=1e-5, negative=3, workers=3,sg=1) # default = 1 worker = no parallelizationd')


# In[26]:


cosine_similarity1 = numpy.dot(model['ဗိုလ်ကြီး'], model['စစ်ပြန်'])/(numpy.linalg.norm(model['ဗိုလ်ကြီး'])* numpy.linalg.norm(model['စစ်ပြန်']))


# In[165]:


model2.wv.cosine_similarities('စစ်ပြန်','ဗိုလ်ကြီး')


# In[13]:


model.predict_output_word('နုပျို')


# In[27]:


print ("Cosine Similarity: ",cosine_similarity1)


# In[14]:


print(model.similarity('နုပျို','တက်ကြွ'))


# In[465]:


x = model['စာချုပ်']


# In[466]:


type(x)


# In[16]:


model.most_similar('ဗိုလ်ကြီး')


# In[66]:


model.predict_output_word('မျက်လုံး')


# In[67]:


model.predict_output_word('တိုင်းပြည်')


# In[33]:


model.most_similar('တိုင်းပြည်')


# In[15]:


model.most_similar('ဂျပန်')


# In[16]:


model.most_similar('ညီညွတ်')


# In[17]:


model.most_similar('စိတ်ဓာတ်')


# In[18]:


model.most_similar('အပြောင်းအလဲ')


# In[34]:


model.most_similar(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[38]:


model.most_similar(positive=[ 'အမည်','အောင်ဆန်း',], negative=['ဂျပန်'])


# In[39]:


model.most_similar(positive=[ 'အမည်','ဂျပန်'], negative=['အောင်ဆန်း'])


# In[35]:


model.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# # Build Word2Vec Model with CBOW include negative 3
# Not Good

# In[12]:


get_ipython().run_cell_magic('time', '', 'model1 = Word2Vec(texts, size=100, window=5, min_count=5,sample=1e-5, negative=3, workers=3,sg=0) # default = 1 worker = no parallelizationd')


# In[20]:


model1.most_similar('ဗိုလ်ကြီး')


# In[21]:


model1.most_similar('ညီညွတ်')


# In[22]:


model1.most_similar('ဂျပန်')


# In[23]:


model1.most_similar('အပြောင်းအလဲ')


# # Build Word2Vec Model with FastText(CBOW) include ngeative 3
# Good but Not Recommand

# In[13]:


get_ipython().run_cell_magic('time', '', 'model_Fast_Cbow = FastText(texts, size=100, window=5, min_count=5,sample=1e-5, negative=3, workers=3,sg=0) # default = 1 worker = no parallelizationd')


# In[25]:


model_Fast_Cbow.most_similar('ဗိုလ်ကြီး')


# In[26]:


model_Fast_Cbow.most_similar('ဂျပန်')


# In[27]:


model_Fast_Cbow.most_similar('ညီညွတ်')


# # Build Word2Vec Model with FastText(SG) include ngeative 3
# Good but Not Recommand

# In[14]:


get_ipython().run_cell_magic('time', '', 'model_Fast_s = FastText(texts, size=100, window=10, min_count=5,sample=1e-5, negative=3, workers=3,sg=1) # default = 1 worker = no parallelizationd')


# In[29]:


model_Fast_s.most_similar('ဗိုလ်ကြီး')


# In[30]:


model_Fast_s.most_similar('ဂျပန်')


# In[31]:


model_Fast_s.most_similar('ညီညွတ်')


# # Build Word2Vec Model with Skip Gram (Neg=0,Iter=2,size=100,window=5)
# Good and Recommand <3 <3

# In[15]:


get_ipython().run_cell_magic('time', '', 'model2=Word2Vec(texts, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, iter=2, null_word=0, trim_rule=None, sorted_vocab=1)')


# In[41]:


model2.most_similar('ဂျပန်')


# In[42]:


model2.most_similar('ညီညွတ်')


# In[43]:


model2.most_similar('ဗိုလ်ကြီး')


# In[44]:


model2.most_similar('စစ်ပြန်')


# # Build Word2Vec Model with Skip Gram (Neg=0,Iter=2,size=100,window=10)
# Very Good and Recommended 

# In[16]:


get_ipython().run_cell_magic('time', '', 'model3=Word2Vec(texts, size=100, alpha=0.025, window=10, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, iter=2, null_word=0, trim_rule=None, sorted_vocab=1)')


# # Word2Vec Parameter Documentation

# text         = data |
# size         = Dimensionality of the word vectors |
# alpha        = The initial learning rate |
# window       = Maximum distance between the current and predicted word within a sentence |
# min_count    = Ignores all words with total frequency lower than this |
# seed         = for the random number generator |
# worker       = Use these many worker threads to train the model (=faster training with multicore machines) |
# min_alpha    = Learning rate will linearly drop to min_alpha as training progresses |
# sg           = Training algorithm: 1 for skip-gram; otherwise CBOW |
# hs           = If 1, hierarchical softmax will be used for model training. If 0,negative sampling will be used |
# iter         = Number of iterations (epochs) over the corpus |
# sorted_vocab = If 1, sort the vocabulary by descending frequency before assigning word indexes |

# In[76]:


from pprint import pprint
pprint (model3.wv.vocab)


# In[77]:


print(len((model3.wv.vocab)))


# In[118]:


w2v = dict(zip(model3.wv.index2word, model.wv.syn0))


# In[125]:


print (type(w2v))


# # Calculation Cosine Similarity

# In[19]:


import numpy


# # The Formula of Cosine Similarity 
# (click down cell link)

# https://wikimedia.org/api/rest_v1/media/math/render/svg/1d94e5903f7936d3c131e040ef2c51b473dd071d

# https://wikimedia.org/api/rest_v1/media/math/render/svg/fb9fc371e46e02d0ef51e781e7397629425856b5

# In[20]:


cosine_similarity = numpy.dot(model3['ဗိုလ်ကြီး'], model3['စစ်ပြန်'])/(numpy.linalg.norm(model3['ဗိုလ်ကြီး'])* numpy.linalg.norm(model3['စစ်ပြန်']))


# In[21]:


print ("Cosine Similarity: ",cosine_similarity)


# In[24]:


import math as _math


# In[25]:


log2 =_math.log(numpy.dot(model3['ဗိုလ်ကြီး'], model3['စစ်ပြန်'])/(numpy.linalg.norm(model3['ဗိုလ်ကြီး'])* numpy.linalg.norm(model3['စစ်ပြန်'])))


# In[149]:


print (log2)


# # -----------------------------*****------------------------------------

# # Corpora

# In[159]:


from gensim import corpora
import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()


# In[160]:


dictionary = corpora.Dictionary(texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
print(dictionary)


# Here we assigned a unique integer ID to all words appearing in the processed corpus with the [gensim.corpora.dictionary.Dictionary](https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary) class. This sweeps across the texts, collecting word counts and relevant statistics. In the end, we see there are so many distinct words in the processed corpus, which means each document will be represented by number. To see the mapping between words and their ids:

# In[161]:


print(dictionary.token2id)


# In[117]:


model3.most_similar(positive=['ကမ်းရိုး'], negative=['ကုန်တင်သင်္ဘော'])


# In[111]:


model3.batch_words


# In[114]:


model3.compute_loss


# In[115]:


model3.corpus_count


# In[75]:


model3.most_similar('ဥပဒေ')


# In[46]:


model3.most_similar('ဗိုလ်ကြီး')


# In[47]:


model3.most_similar('ညီညွတ်')


# In[48]:


model3.most_similar('စစ်ပြန်')


# In[49]:


model3.most_similar('ဂျပန်')


# In[51]:


model3.most_similar('လုံခြုံ')


# In[57]:


model3.most_similar('အမျိုးသား')


# In[50]:


model3.most_similar('မျက်လုံး')


# In[136]:


model3.similar_by_word('အိုမင်း')


# In[135]:


model3.similarity('ဗိုလ်ကြီး','ကြက်မောက်သီး')


# In[467]:


from __future__ import absolute_import, division, print_function
import numpy as np
count = 10000
word_vectors_matrix = np.ndarray(shape=(count, 100), dtype='float64')
word_list = []
i = 0
for word in model3.wv.vocab:
    word_vectors_matrix[i] = model3[word]
    t = word_vectors_matrix[i]
    print(t)
    word_list.append(word)
    i = i+1
    if i == count:
        break
print("word_vectors_matrix shape is ", word_vectors_matrix.shape)


# In[468]:


from __future__ import absolute_import, division, print_function
import numpy as np
count = 10000
word_vectors_matrix = np.ndarray(shape=(count, 100), dtype='float64')
word_list = []
i = 0
for word in model3.wv.vocab:
    word_vectors_matrix[i] = model3[word]
    print(model3[word])
    #t = word_vectors_matrix[i]
    #print(t)
    word_list.append(word)
    i = i+1
    if i == count:
        break
print("word_vectors_matrix shape is ", word_vectors_matrix.shape)


# In[80]:


from __future__ import absolute_import, division, print_function
import numpy as np
count = 10000
word_vectors_matrix = np.ndarray(shape=(count, 100), dtype='float64')
word_list = []
i = 0
for word in model3.wv.vocab:
    word_vectors_matrix[i] = model3[word]
    
    word_list.append(word)
    i = i+1
    if i == count:
        break
print("word_vectors_matrix shape is ", word_vectors_matrix.shape)


# In[81]:


import sklearn.manifold
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)
print("word_vectors_matrix_2d shape is ", word_vectors_matrix_2d.shape)


# In[82]:


import pandas as pd
points = pd.DataFrame(
    [
        (word, coords[0], coords[1]) 
        for word, coords in [
            (word, word_vectors_matrix_2d[word_list.index(word)])
            for word in word_list
        ]
    ],
    columns=["word", "x", "y"]
)
print("Points DataFrame built")


# In[92]:


points.head(30)


# In[84]:


get_ipython().run_cell_magic('time', '', 'import matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline\nsns.set_context("poster")')


# In[85]:


points.plot.scatter("x", "y", s=10, figsize=(20, 12))


# In[99]:


get_ipython().run_cell_magic('time', '', 'def plot_region(x_bounds, y_bounds):\n    slice = points[\n        (x_bounds[0] <= points.x) &\n        (points.x <= x_bounds[1]) &\n        (y_bounds[0] <= points.y) &\n        (points.y <= y_bounds[1]) \n    ]\n    \n    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))\n    for i, point in slice.iterrows():\n        #print(point.word)\n        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)')


# In[101]:


get_ipython().run_cell_magic('time', '', 'plot_region(x_bounds=(25, 30), y_bounds=(25, 30))')


# # Build Word2Vec Model with Skip Gram (Neg=0,Iter=2,size=100,window=15)
# Very Good and Recommand <3 <3

# In[17]:


get_ipython().run_cell_magic('time', '', 'model4=Word2Vec(texts, size=100, alpha=0.025, window=15, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, iter=2, null_word=0, trim_rule=None, sorted_vocab=1)')


# In[53]:


model4.most_similar('စစ်ပြန်')


# In[54]:


model4.most_similar('ဂျပန်')


# In[55]:


model4.most_similar('လုံခြုံ')


# In[56]:


model4.most_similar('ညီညွတ်')


#  # Build Word2Vec Model with Skip Gram (Neg=0,Iter=2,size=100,window=3)

# In[18]:


get_ipython().run_cell_magic('time', '', 'model_w3=Word2Vec(texts, size=100, alpha=0.025, window=3, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, iter=2, null_word=0, trim_rule=None, sorted_vocab=1)')


# In[138]:


model_w3.most_similar('စစ်ပြန်')


# In[139]:


model_w3.most_similar('ညီညွတ်')


# In[140]:


model_w3.most_similar('ဂျပန်')


# In[141]:


model_w3.most_similar('လုံခြုံ')


# In[142]:


model_w3.similar_by_word('အိုမင်း')


# # =====================================================

# In[19]:


model.most_similar('ရန်ကုန်')


# In[20]:


model1.most_similar('ရန်ကုန်')


# In[29]:


model_Fast_Cbow.most_similar('ရန်ကုန်')


# In[28]:


model_Fast_s.most_similar('ရန်ကုန်')


# In[24]:


model2.most_similar('ရန်ကုန်')


# In[25]:


model3.most_similar('ရန်ကုန်')


# In[26]:


model4.most_similar('ရန်ကုန်')


# In[27]:


model_w3.most_similar('ရန်ကုန်')


# In[30]:


model.most_similar('ယောက်ျား')


# In[31]:


model1.most_similar('ယောက်ျား')


# In[32]:


model2.most_similar('ယောက်ျား')


# In[33]:


model3.most_similar('ယောက်ျား')


# In[34]:


model4.most_similar('ယောက်ျား')


# In[35]:


model_w3.most_similar('ယောက်ျား')


# In[53]:


model2.score('အောင်မြင်')


# In[88]:


"နာရီ ဘယ် မှာ ဝယ် လို့ ရ နိုင် မလဲ"


# In[87]:


model2.doesnt_match("နာရီ ဘယ် မှာ မြန်မာ ဝယ် လို့ ရ နိုင် မလဲ".split())


# In[89]:


model.doesnt_match("နာရီ ဘယ် မှာ မြန်မာ ဝယ် လို့ ရ နိုင် မလဲ".split())


# In[90]:


model1.doesnt_match("နာရီ ဘယ် မှာ မြန်မာ ဝယ် လို့ ရ နိုင် မလဲ".split())


# In[91]:


model3.doesnt_match("နာရီ ဘယ် မှာ မြန်မာ ဝယ် လို့ ရ နိုင် မလဲ".split())


# In[92]:


model4.doesnt_match("နာရီ ဘယ် မှာ မြန်မာ ဝယ် လို့ ရ နိုင် မလဲ".split())


# In[116]:


model_w3.doesnt_match("မြန်မာ နာရီ ဘယ် မှာ ဝယ် လို့ ရ နိုင် မလဲ".split())


# In[94]:


model.doesnt_match('ထို အစည်းအရုံး သည် အမျိုးသား ရေး စိတ်ဓာတ် ပြင်းပြ ၍ နုပျို တက်ကြွ သည့် အခြေခံ လက္ခဏာ ရှိ ပြီး အစည်းအရုံး နာရီ  ခေါင်းဆောင် များ သည် သက်ကြီး နိုင်ငံရေးသမား များ နှင့် မ တူ တမူထူးခြား ကြ ပါ သည် ။'.split())


# In[95]:


model1.doesnt_match('ထို အစည်းအရုံး သည် အမျိုးသား ရေး စိတ်ဓာတ် ပြင်းပြ ၍ နုပျို တက်ကြွ သည့် အခြေခံ လက္ခဏာ ရှိ ပြီး အစည်းအရုံး နာရီ  ခေါင်းဆောင် များ သည် သက်ကြီး နိုင်ငံရေးသမား များ နှင့် မ တူ တမူထူးခြား ကြ ပါ သည် ။'.split())


# In[96]:


model2.doesnt_match('ထို အစည်းအရုံး သည် အမျိုးသား ရေး စိတ်ဓာတ် ပြင်းပြ ၍ နုပျို တက်ကြွ သည့် အခြေခံ လက္ခဏာ ရှိ ပြီး အစည်းအရုံး နာရီ  ခေါင်းဆောင် များ သည် သက်ကြီး နိုင်ငံရေးသမား များ နှင့် မ တူ တမူထူးခြား ကြ ပါ သည် ။'.split())


# In[97]:


model3.doesnt_match('ထို အစည်းအရုံး သည် အမျိုးသား ရေး စိတ်ဓာတ် ပြင်းပြ ၍ နုပျို တက်ကြွ သည့် အခြေခံ လက္ခဏာ ရှိ ပြီး အစည်းအရုံး နာရီ  ခေါင်းဆောင် များ သည် သက်ကြီး နိုင်ငံရေးသမား များ နှင့် မ တူ တမူထူးခြား ကြ ပါ သည် ။'.split())


# In[98]:


model4.doesnt_match('ထို အစည်းအရုံး သည် အမျိုးသား ရေး စိတ်ဓာတ် ပြင်းပြ ၍ နုပျို တက်ကြွ သည့် အခြေခံ လက္ခဏာ ရှိ ပြီး အစည်းအရုံး နာရီ  ခေါင်းဆောင် များ သည် သက်ကြီး နိုင်ငံရေးသမား များ နှင့် မ တူ တမူထူးခြား ကြ ပါ သည် ။'.split())


# In[99]:


model_w3.doesnt_match('ထို အစည်းအရုံး သည် အမျိုးသား ရေး စိတ်ဓာတ် ပြင်းပြ ၍ နုပျို တက်ကြွ သည့် အခြေခံ လက္ခဏာ ရှိ ပြီး အစည်းအရုံး နာရီ  ခေါင်းဆောင် များ သည် သက်ကြီး နိုင်ငံရေးသမား များ နှင့် မ တူ တမူထူးခြား ကြ ပါ သည် ။'.split())


# In[117]:


model.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[118]:


model1.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[119]:


model2.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[120]:


model3.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[121]:


model4.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[122]:


model_w3.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[299]:


r = model3['ဘုရင်'] - model3['ယောက်ျား'] + model3['မိန်းမ']
model3.similar_by_vector(r)


# In[303]:


model.most_similar_cosmul(['ယောက်ျား','မိန်းမ','အချစ်'])


# In[302]:


model2.most_similar_cosmul(['ယောက်ျား','မိန်းမ','အချစ်'])


# In[301]:


model3.most_similar_cosmul(['ယောက်ျား','မိန်းမ','အချစ်'])


# In[304]:


model4.most_similar_cosmul(['ယောက်ျား','မိန်းမ','အချစ်'])


# In[305]:


model_w3.most_similar_cosmul(['ယောက်ျား','မိန်းမ','အချစ်'])


# In[361]:


model.most_similar_cosmul(['အဖိုး','အဖွား','မြေး'])


# In[362]:


model2.most_similar_cosmul(['အဖိုး','အဖွား','မြေး'])


# In[363]:


model3.most_similar_cosmul(['အဖိုး','အဖွား','မြေး'])


# In[364]:


model4.most_similar_cosmul(['အဖိုး','အဖွား','မြေး'])


# In[365]:


model_w3.most_similar_cosmul(['အဖိုး','အဖွား','မြေး'])


# In[311]:


model.most_similar_cosmul(['ဆေးရုံ','ဆေးခန်း','လူနာ'])


# In[312]:


model2.most_similar_cosmul(['ဆေးရုံ','ဆေးခန်း','လူနာ'])


# In[313]:


model3.most_similar_cosmul(['ဆေးရုံ','ဆေးခန်း','လူနာ'])


# In[315]:


model4.most_similar_cosmul(['ဆေးရုံ','ဆေးခန်း','လူနာ'])


# In[316]:


model_w3.most_similar_cosmul(['ဆေးရုံ','ဆေးခန်း','လူနာ'])


# In[295]:


model.most_similar_cosmul(positive=['မိန်းမ','ဘုရင်' ], negative=['ယောက်ျား'])


# In[296]:


model2.most_similar_cosmul(positive=['မိန်းမ','ဘုရင်' ], negative=['ယောက်ျား'])


# In[156]:


model3.most_similar_cosmul(positive=['မိန်းမ','ဘုရင်' ], negative=['ယောက်ျား'])


# In[460]:


model4.most_similar_cosmul(positive=['မိန်းမ','ဘုရင်မ' ], negative=['ယောက်ျား'])


# In[167]:


model_w3.most_similar_cosmul(positive=['မိန်းမ','ဘုရင်' ], negative=['ယောက်ျား'])


# In[339]:


model.most_similar_cosmul(positive=['စစ်သား','ဗိုလ်ချုပ်'], negative=['စစ်တပ်'])


# In[340]:


model2.most_similar_cosmul(positive=['စစ်သား','ဗိုလ်ချုပ်'], negative=['စစ်တပ်'])


# In[341]:


model3.most_similar_cosmul(positive=['စစ်သား','ဗိုလ်ချုပ်'], negative=['စစ်တပ်'])


# In[342]:


model4.most_similar_cosmul(positive=['စစ်သား','ဗိုလ်ချုပ်'], negative=['စစ်တပ်'])


# In[343]:


model_w3.most_similar_cosmul(positive=['စစ်သား','ဗိုလ်ချုပ်'], negative=['စစ်တပ်'])


# In[224]:


model.most_similar_cosmul(positive=['ဘုရင်','နိုင်ငံ'], negative=['ဘုရင်မ'])


# In[225]:


model2.most_similar_cosmul(positive=['ဘုရင်','နိုင်ငံ'], negative=['ဘုရင်မ'])


# In[226]:


model3.most_similar_cosmul(positive=['ဘုရင်','နိုင်ငံ'], negative=['ဘုရင်မ'])


# In[227]:


model4.most_similar_cosmul(positive=['ဘုရင်','နိုင်ငံ'], negative=['ဘုရင်မ'])


# In[228]:


model_w3.most_similar_cosmul(positive=['ဘုရင်','နိုင်ငံ'], negative=['ဘုရင်မ'])


# In[168]:


model.most_similar_cosmul(positive=['ရန်ကုန်','မြို့' ], negative=['မန္တလေး'])


# In[170]:


model2.most_similar_cosmul(positive=['ရန်ကုန်','မြို့' ], negative=['မန္တလေး'])


# In[171]:


model3.most_similar_cosmul(positive=['ရန်ကုန်','မြို့' ], negative=['မန္တလေး'])


# In[172]:


model4.most_similar_cosmul(positive=['ရန်ကုန်','မြို့' ], negative=['မန္တလေး'])


# In[173]:


model_w3.most_similar_cosmul(positive=['ရန်ကုန်','မြို့' ], negative=['မန္တလေး'])


# In[178]:


model.most_similar_cosmul(positive=['မြန်မာ','နိုင်ငံ' ], negative=['ဂျပန်'])


# In[177]:


model2.most_similar_cosmul(positive=['မြန်မာ','နိုင်ငံ' ], negative=['ဂျပန်'])


# In[179]:


model3.most_similar_cosmul(positive=['မြန်မာ','နိုင်ငံ' ], negative=['ဂျပန်'])


# In[180]:


model4.most_similar_cosmul(positive=['မြန်မာ','နိုင်ငံ' ], negative=['ဂျပန်'])


# In[181]:


model_w3.most_similar_cosmul(positive=['မြန်မာ','နိုင်ငံ' ], negative=['ဂျပန်'])


# In[292]:


model.most_similar_cosmul(positive=['ယောက်ျား','ဘုရင်မ'], negative=['မိန်းမ'])


# In[293]:


model2.most_similar_cosmul(positive=['ယောက်ျား','ဘုရင်မ'], negative=['မိန်းမ'])


# In[290]:


model3.most_similar_cosmul(positive=['ယောက်ျား','ဘုရင်မ'], negative=['မိန်းမ'])


# In[291]:


model4.most_similar_cosmul(positive=['ယောက်ျား','ဘုရင်မ'], negative=['မိန်းမ'])


# In[317]:


model3.most_similar_cosmul(positive=['လူကြီး','အသက်'], negative=['ကလေး'])


# In[263]:


model.most_similar_cosmul(positive=['ပြည်သူ','တိုင်းပြည်' ], negative=['ပြည်သား'])


# In[264]:


model2.most_similar_cosmul(positive=['ပြည်သူ','တိုင်းပြည်' ], negative=['ပြည်သား'])


# In[265]:


model3.most_similar_cosmul(positive=['ပြည်သူ','တိုင်းပြည်' ], negative=['ပြည်သား'])


# In[266]:


model4.most_similar_cosmul(positive=['ပြည်သူ','တိုင်းပြည်' ], negative=['ပြည်သား'])


# In[267]:


model_w3.most_similar_cosmul(positive=['ပြည်သူ','တိုင်းပြည်' ], negative=['ပြည်သား'])


# In[274]:


model.most_similar_cosmul(positive=['အချစ်','မိန်းမ'], negative=['ယောက်ျား'])


# In[320]:


model2.most_similar_cosmul(positive=['ယောက်ျား','မိန်းမ'], negative=['အချစ်'])


# In[284]:


model3.most_similar_cosmul(positive=['ယောက်ျား','မိန်းမ'], negative=['အချစ်'])


# In[278]:


model4.most_similar_cosmul(positive=['ယောက်ျား','မိန်းမ'], negative=['အချစ်'])


# In[279]:


model_w3.most_similar_cosmul(positive=['ယောက်ျား','မိန်းမ'], negative=['အချစ်'])


# In[188]:


model.most_similar_cosmul(positive=[ 'နာနတ်သီး','အသီး' ], negative=['သရက်သီး'])


# In[187]:


model2.most_similar_cosmul(positive=[ 'နာနတ်သီး','အသီး' ], negative=['သရက်သီး'])


# In[185]:


model3.most_similar_cosmul(positive=[ 'နာနတ်သီး','အသီး' ], negative=['သရက်သီး'])


# In[189]:


model4.most_similar_cosmul(positive=[ 'နာနတ်သီး','အသီး' ], negative=['သရက်သီး'])


# In[190]:


model_w3.most_similar_cosmul(positive=[ 'နာနတ်သီး','အသီး' ], negative=['သရက်သီး'])


# In[191]:


model.most_similar_cosmul(positive=[ 'ရန်ကုန်','မြန်မာ' ], negative=['မန္တလေး'])


# In[192]:


model2.most_similar_cosmul(positive=[ 'ရန်ကုန်','မြန်မာ' ], negative=['မန္တလေး'])


# In[193]:


model3.most_similar_cosmul(positive=[ 'ရန်ကုန်','မြန်မာ' ], negative=['မန္တလေး'])


# In[194]:


model4.most_similar_cosmul(positive=[ 'ရန်ကုန်','မြန်မာ' ], negative=['မန္တလေး'])


# In[195]:


model_w3.most_similar_cosmul(positive=[ 'ရန်ကုန်','မြန်မာ' ], negative=['မန္တလေး'])


# In[258]:


model.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# In[259]:


model2.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# In[260]:


model3.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# In[261]:


model4.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# In[262]:


model_w3.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# In[148]:


model.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး သရက်သီး ဒူးရင်းသီး စပျစ်သီး စပါး သစ္စာခံ '.split())


# In[147]:


model1.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး သရက်သီး ဒူးရင်းသီး စပျစ်သီး စပါး သစ္စာခံ '.split())


# In[146]:


model2.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး ဒူးရင်းသီး သရက်သီး စပျစ်သီး စပါး သစ္စာခံ'.split())


# In[154]:


model3.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး ဒူးရင်းသီး သရက်သီး စပျစ်သီး စပါး သစ္စာခံ'.split())


# In[144]:


model4.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး ဒူးရင်းသီး စပျစ်သီး သရက်သီး စပါး သစ္စာခံ '.split())


# In[143]:


model_w3.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး သရက်သီး ဒူးရင်းသီး စပျစ်သီး စပါး သစ္စာခံ'.split())


# In[210]:


model.doesnt_match('စပယ်ပန်း နှင်းဆီ သစ်ခွ မြန်မာ စကားပန်း ပန်းဥယျာဉ်'.split())


# In[326]:


model2.doesnt_match('စပယ်ပန်း နှင်းဆီပန်း မြန်မာ စကားပန်း ပန်း ပန်းဥယျာဉ် '.split())


# In[323]:


model3.doesnt_match('စပယ်ပန်း စကားပန်း ပန်း ပန်းဥယျာဉ် မြန်မာ နှင်းဆီပန်း'.split())


# In[324]:


model4.doesnt_match('စပယ်ပန်း မြန်မာ စကားပန်း ပန်း ပန်းဥယျာဉ် နှင်းဆီပန်း'.split())


# In[325]:


model_w3.doesnt_match('စပယ်ပန်း စကားပန်း ပန်း  ပန်းဥယျာဉ် မြန်မာ နှင်းဆီပန်း'.split())


# In[327]:


model.doesnt_match('မင်းသား မင်းသမီး ရုပ်ရှင်ကား ဇတ်ကား ဆရာဝန် '.split())


# In[328]:


model2.doesnt_match('မင်းသား မင်းသမီး ရုပ်ရှင်ကား ဇတ်ကား ဆရာဝန် '.split())


# In[329]:


model3.doesnt_match('မင်းသား မင်းသမီး ရုပ်ရှင်ကား ဇတ်ကား ဆရာဝန် '.split())


# In[330]:


model4.doesnt_match('မင်းသား မင်းသမီး ရုပ်ရှင်ကား ဇတ်ကား ဆရာဝန် '.split())


# In[331]:


model_w3.doesnt_match('မင်းသား မင်းသမီး ရုပ်ရှင်ကား ဇတ်ကား ဆရာဝန် '.split())


# In[344]:


get_ipython().run_cell_magic('time', '', 'model_n5 = Word2Vec(texts, size=100, window=5, min_count=5,sample=1e-5, negative=3, workers=3,sg=1) # default = 1 worker = no parallelizationd')


# In[345]:


model_n5.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး သရက်သီး ဒူးရင်းသီး စပျစ်သီး စပါး သစ္စာခံ '.split())


# In[346]:


model_n5.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# # SkipGram ,window 10,iter=3 

# In[390]:


get_ipython().run_cell_magic('time', '', 'model_n10_iter3=Word2Vec(texts, size=100, alpha=0.025, window=10, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, iter=10, null_word=0, trim_rule=None, sorted_vocab=1)')


# In[448]:


model.wv.distance('မိန်းမ','ဘုရင်မ')


# In[449]:


model2.wv.distance('မိန်းမ','ဘုရင်မ')


# In[450]:


model3.wv.distance('မိန်းမ','ဘုရင်မ')


# In[459]:


model4.wv.distance('မိန်းမ','ဘုရင်မ')


# In[452]:


model_w3.wv.distance('မိန်းမ','ဘုရင်မ')


# In[453]:


model_n10_iter3.wv.distance('မိန်းမ','ဘုရင်မ')


# In[401]:


model_n10_iter3.wv.most_similar_cosmul(positive=['ကျောင်းသူ','တက္ကသိုလ်' ], negative=['ကျောင်းသား'])


# In[392]:


model_n10_iter3.most_similar_cosmul(positive=[ 'ရန်ကုန်','မြန်မာ' ], negative=['မန္တလေး'])


# In[393]:


model_n10_iter3.most_similar_cosmul(positive=['ယောက်ျား','မိန်းမ'], negative=['အချစ်'])


# In[394]:


model_n10_iter3.most_similar_cosmul(positive=['စစ်သား','ဗိုလ်ချုပ်'], negative=['စစ်တပ်'])


# In[395]:


model_n10_iter3.most_similar_cosmul(positive=['ယောက်ျား','ဘုရင်မ'], negative=['မိန်းမ'])


# In[396]:


model_n10_iter3.most_similar_cosmul(positive=['ယောက်ျား','ဘုရင်'], negative=['မိန်းမ'])


# In[397]:


model_n10_iter3.most_similar_cosmul(['အဖိုး','အဖွား','မြေး'])


# In[398]:


model_n10_iter3.most_similar_cosmul(positive=['မြန်မာ','နိုင်ငံ' ], negative=['ဂျပန်'])


# In[399]:


model_n10_iter3.most_similar_cosmul(positive=['အောင်ဆန်း', 'အမည်'], negative=['ဂျပန်'])


# In[389]:


model_n10_iter3.doesnt_match('သစ်သီး ပန်းသီးပင် နာနတ်သီး သရက်သီး ဒူးရင်းသီး စပျစ်သီး စပါး သစ္စာခံ '.split())


# In[66]:


from gensim.similarities import WmdSimilarity


# In[67]:


instance = WmdSimilarity(texts, model, num_best=10)


# In[68]:


print (instance)


# In[69]:


instance


# In[79]:


from time import time
from nltk import word_tokenize


# In[77]:


def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


# In[ ]:


print 'Cell took %.2f seconds to run.' %(time() - start)

