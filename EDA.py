#dependencies

import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.patches as mpatches

#import train data
train = pickle.load(open("train_data","rb"))

train_cat_freq = []

category = list(set(train['category'].tolist()))

for i in category:
   train_cat_freq.append(train['category'].tolist().count(i))

#uncomment this section to view frequency distribution of documents
##plt.bar(category,train_cat_freq)
##plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(train.texts).toarray()
print(len(features[1]) == len(features[0]))
labels = train.category

model_tsne = TSNE(n_components=2, verbose = 1, perplexity = 200,
                  learning_rate = 30, random_state=0)

tsne_result = model_tsne.fit_transform(features)
print(tsne_result)
columns = ['X','Y','labels','colors']
df = pd.DataFrame(columns = columns)
df['X'] = tsne_result[:,0]
df['Y'] = tsne_result[:,1]
df['labels'] = labels

colors = ["g","r","y","b",'c']
for i in range(len(df['labels'])):
    df['colors'][i] = colors[df['labels'][i]-1]


gpatch = mpatches.Patch(color='g', label='Business')
rpatch = mpatches.Patch(color='r', label='Entertainment')
ypatch = mpatches.Patch(color='y', label='Politics')
bpatch = mpatches.Patch(color='b', label='Sports')
cpatch = mpatches.Patch(color='c', label='Technology')

plt.legend(handles=[gpatch,rpatch,ypatch,bpatch,cpatch])

cols = [i for i in df['colors']]    
plt.scatter(df['X'],df['Y'],c= cols)
plt.show()
