
# coding: utf-8

# # musiXmatch dataset - list of matches
#      matches provided by musiXmatch based on artist names
#      and song titles from the Million Song Dataset.
#  MORE INFORMATION:
#      http://labrosa.ee.columbia.edu/millionsong/musixmatch
#  FORMAT:
#      -> comment, ignore
#      tid|artist name|title|mxm tid|artist_name|title
#         tid          -> Million Song Dataset track ID
#         artist name  -> artist name in the MSD
#         title        -> title in the MSD
#         mxm tid      -> musiXmatch track ID
#         artist name  -> artist name for mXm
#         title        -> title for mXm
#         |            -> actual separator: <SEP>
# 

# In[1]:


import pandas as pd

def load_details():
    with open('/media/clayton/MyFiles/Projects/Datasets/mxm_songs.txt', 'r') as all_songs:
        x = all_songs.readlines()

        s = []

        for song in x:
            data = song.split('<SEP>')
            s.append(data)
        df = pd.DataFrame(data=s)
        df.columns = ['tid', 'artist name', 'title', 'mxm tid', 'artist_name', 'title tid']
        df = df.set_index('tid')
        return df

song_details = load_details()


# # Creating a list of term frequency for each song (sample)

# In[2]:


def load_samples():
    with open('/media/clayton/MyFiles/Projects/Datasets/mxm_samples.txt', 'r') as sample:
        return sample.readlines()
   
lyrics = []
song_ids = []
samples = load_samples()

with open('/media/clayton/MyFiles/Projects/Datasets/mxm_bag_of_words.txt', 'r') as file:
    bag_of_words = ''.join(file.readlines()).split(',')

for i in range(len(samples)):
    l = samples[i].split(',')
    song_ids.append(l[0])
    tf = []
    for j in range(2, len(l)):        
        word_index = int(l[j].split(':')[0])-1
        frequency = int(l[j].split(':')[1])
        for k in range(int(frequency)):
            tf.append(bag_of_words[word_index])      
    lyrics.append(' '.join(tf))    


# # Sample demo

# In[3]:


lyrics[0]


# # TF-IDF

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, stop_words='english')

tfidf_matrix = tfidf.fit_transform(lyrics)

print(tfidf.get_feature_names())
# print(tfidf_matrix.toarray())


# # Cosine Similarity

# In[5]:


#Implement some way to visualize the data outcome

# for i in range(len(lyrics)):
#     song1_id = song_ids[i]
#     artist1_name = song_details.loc[song1_id][0]
#     song1_name = song_details.loc[song1_id][1]
    
#     for j in range(len(lyrics)):
#         cos_sim = cosine_similarity(tfidf_matrix[i].toarray(), tfidf_matrix[j].toarray())[0][0]
        
#         song2_id = song_ids[j]
#         artist2_name = song_details.loc[song2_id][0]
#         song2_name = song_details.loc[song2_id][1]
        
#         if (cos_sim < 1 and cos_sim >= 0.5):
#             print('Song \'{}-{}\' is {} similar to song \'{}-{}\' '.format(
#                 artist1_name, song1_name, cos_sim, artist2_name, song2_name))


# In[ ]:




