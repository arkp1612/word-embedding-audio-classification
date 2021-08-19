import urllib, tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import re


def word_embedder():
    req = urllib.request.Request('http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()

    words = the_page.replace(b'\t',b'\n').split(b'\n')


    freq = words[1:len(words):2]
    words = words[0:(len(words)+1):2]

    words = [words[i].decode('utf-8').lower() for i in range(len(words))]


    all_tags = ['rock', 'pop', 'indie', 'electronic', 'dance', 'alternative rock', 
    'jazz', 'singer songwriter', 'metal', 'chillout', 'classic rock', 
    'soul', 'indie rock', 'electronica', 'folk', 'instrumental', 
    'punk', 'oldies', 'mellow', 'sexy', 'loved', 'sad', 'happy', 
    'good', 'romantic', 'melancholic', 'great', 'dark', 'dreamy','hot', 
    'energetic', 'calm', 'funny', 'haunting', 'intense', 'alternative', 
    'beautiful', 'awesome', 'british', 'chill', 'american', 'cool', 
    'favorite', 'acoustic', 'party', '2000s', '80s', '90s', '60s', '70s']


    req = urllib.request.Request('https://raw.githubusercontent.com/arkp1612/word-embedding-audio-classification/main/sensible_tags_encoding.txt')
    with urllib.request.urlopen(req) as response:
         the_page = response.read()

    all_sensible_tags = the_page.split(b'\n')

    all_sensible_tags = [x.decode('utf-8').split('\t') for x in all_sensible_tags]

    k = [all_sensible_tags[i][0] for i in range(len(all_sensible_tags))]
    v = [all_sensible_tags[i][-1] for i in range(len(all_sensible_tags))]

    k.append(':00s')
    v.append('2000s')

    sensible_pd = pd.read_csv('sensible_tags_encoding_V1.txt',sep='	',header=None)
    sensible_tags = list(np.array(sensible_pd[1]))
    sensible_tags = np.unique(sensible_tags)

    sensible_tokens = [re.sub(r'[^A-Za-z0-9]+', ' ', x) for x in sensible_tags] 
    stripped_tags = [x.strip() for x in sensible_tokens]
    stripped_tags = np.unique(stripped_tags)
    stripped_tags = stripped_tags[1:]


    with tf.Graph().as_default():
        embed = hub.load("https://tfhub.dev/google/nnlm-en-dim128/2")     #https://tfhub.dev/google/Wiki-words-250/2 #https://tfhub.dev/google/Wiki-words-500/2
        embeddings = embed(stripped_tags)
        with tf.compat.v1.train.MonitoredSession() as sess:
            values = sess.run(embeddings)

    data = dict(zip(stripped_tags, values))


    def get_neighbors(all_data, target, count=3):
      # Compute points in all_data that are the closest to "target".
      # Sort the words based on the length of the vector between them.
      # Then pick "count" closest ones.
      return sorted(all_data, key=lambda emb_key: np.linalg.norm(all_data[emb_key] - target))[:count]


    top_50_dict = { your_key: data[your_key] for your_key in all_tags }

    # calcualtion of thresholds for dictionary
    # import scipy.stats
    # for x in all_tags:
    #     print('"', x,'"',':',scipy.stats.norm.ppf(0.001,all_distances_pd[x].mean(),all_distances_pd[x].std()),',')# calcualtion of thresholds for dictionary
    threshold_dict_tag = {"rock" : 0.8941884313329378 ,
    "pop" : 0.9195923962858048 ,
    "indie" : 0.8913482259168488 ,
    "electronic" : 1.0478442809494433 ,
    "dance" : 0.9261400109868414 ,
    "alternative rock" : 0.9475134977058159 ,
    "jazz" : 0.8593375099732143 ,
    "singer songwriter" : 1.149694470691324 ,
    "metal" : 0.9642991137546975 ,
    "chillout" : 0.9670640819899192 ,
    "classic rock" : 0.8658979624989583 ,
    "soul" : 0.9309216422303255 ,
    "indie rock" : 0.9282593848164515 ,
    "electronica" : 0.8775971090604648 ,
    "folk" : 0.9185292354131764 ,
    "instrumental" : 1.0675786328877708 ,
    "punk" : 0.8416480659554245 ,
    "oldies" : 0.9738604490331451 ,
    "mellow" : 0.8680267835229767 ,
    "sexy" : 0.9014923905814844 ,
    "loved" : 1.0028553558891369 ,
    "sad" : 0.9856225434438972 ,
    "happy" : 1.0239110438439953 ,
    "good" : 0.9517758727358585 ,
    "romantic" : 0.9461531385197661 ,
    "melancholic" : 0.9056599036152433 ,
    "great" : 0.9526484141348239 ,
    "dark" : 0.9784131878055917 ,
    "dreamy" : 0.8898196034464734 ,
    "hot" : 1.0072853030210123 ,
    "energetic" : 0.9617915171262403 ,
    "calm" : 1.0442804716775644 ,
    "funny" : 0.9135906598795938 ,
    "haunting" : 0.9479750555108033 ,
    "intense" : 1.0000263710153725 ,
    "alternative" : 1.0853989459358326 ,
    "beautiful" : 0.9404607757702008 ,
    "awesome" : 0.9088700355939647 ,
    "british" : 0.9862536342592774 ,
    "chill" : 1.0168272335638537 ,
    "american" : 1.0008868770738542 ,
    "cool" : 0.9088534396749302 ,
    "favorite" : 1.0102748896921583 ,
    "acoustic" : 0.8905005028983665 ,
    "party" : 1.106223356832784 ,
    "2000s" : 1.170885653133619 ,
    "80s" : 1.0712495378404536 ,
    "90s" : 1.0873825367293513 ,
    "60s" : 1.1275430451883692 ,
    "70s" : 1.0968716180136784 ,}


    tag = []
    word_embeded_tag = []
    distance_list = []

    for i in range(len(stripped_tags)):
        closest_sensible_tag = get_neighbors(top_50_dict,data[stripped_tags[i]],1)
        distance = np.linalg.norm(data[stripped_tags[i]]-data[closest_sensible_tag[0]])
        if distance <= threshold_dict_tag[closest_sensible_tag[0]]:
            tag.append(stripped_tags[i])
            word_embeded_tag.append(closest_sensible_tag[0])
    #        distance_list.append(str(distance))
        else:
            tag.append(stripped_tags[i])
            word_embeded_tag.append(stripped_tags[i])

    a = list(zip(tag,word_embeded_tag))
    k_1 = [x[0] for x in a]
    v_1 = [x[-1] for x in a]

    k_new = []
    v_new = []

    for i in range(len(k_1)):
        if (((k_1[i]!=v_1[i])|(v_1[i] in all_tags))|(k_1[i] in all_tags))&(k_1[i] not in res):
            k_new.append(k_1[i])
            v_new.append(v_1[i])

    k_final = k + k_new
    v_final = v + v_new
    l = list(zip(k_final,v_final))

    with open("word_embedding_encoding.txt", 'w',encoding="utf-8") as output:
        for row in l:
            output.write(str(row[0]) + '\t' + str(row[1]) + '\n')