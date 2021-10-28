#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def tag_filtering(score_level):
    import requests, json
    import re
    import numpy as np 
    import urllib
    
    
    req = urllib.request.Request('http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
        
    words = the_page.replace(b'\t',b'\n').split(b'\n')
    words = words[0:(len(words)+1):2]
    
    req = urllib.request.Request('https://gist.githubusercontent.com/sampsyo/1241307/raw/208ab2e4b5b576ebc51d801b039f93ee2bbc33ea/genres.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    
    genres = the_page.split(b'\n')
    
    
    req = urllib.request.Request('https://raw.githubusercontent.com/taikuukaits/SimpleWordlists/master/Wordlist-Adjectives-All.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    
    descriptives = the_page.split(b'\n')
    
    
    
    url = 'https://raw.githubusercontent.com/dariusk/corpora/master/data/humans/moods.json'
    resp = requests.get(url)
    data = json.loads(resp.text)
    
    def decoder(x):
        return x.decode('utf-8')
    
    moods = data['moods']
    words = [x.lower() for x in list(map(decoder,words))]
    descriptives = [x.lower() for x in list(map(decoder,descriptives))]
    genres = [x.lower() for x in list(map(decoder,genres))]
    
    labels = []
    for i in range(len(words)):
        if words[i] in (gen+genres) :
            labels.append('genres')

        elif l.find(words[i])!=-1:
            labels.append('genres')
        
        elif words[i] in moods:
            labels.append('moods')
    
        elif words[i] in descriptives:
            labels.append('descriptive')
    
        elif sum([str(x) in words[i] for x in range(10)])!=0:
            labels.append('number')
        else:
            labels.append('junk')
            
            
    resp =  urllib.request.urlopen('https://musicbrainz.org/genres')
    html_bytes = resp.read()
    html = html_bytes.decode("utf-8")
    gen =  re.split('<bdi>|</bdi>', html[9067:78729])[1::2]
        
    junk =[keys for keys in w_l.keys() if w_l[keys] == 'junk']
    genres = gen+genres
        
    step_2_cleaning_genres_list = []

    for i in range(len(junk)):
        for j in range(len(genres)):
            if junk[i].find(genres[j])!=-1:
                step_2_cleaning_genres_list.append(junk[i])
                break
            else:
                continue
                    
       
    junk_2 = np.setdiff1d(junk,step_2_cleaning_genres_list)
    
    step_2_cleaning_moods_list = []

    for i in range(len(junk_2)):
        for j in range(len(moods)):
            if junk_2[i].find(moods[j])!=-1:
                step_2_cleaning_moods_list.append(junk_2[i])
                break
            else:
                continue
                
    junk_3 = np.setdiff1d(junk_2,step_2_cleaning_moods_list)
    descriptives = descriptives[0:len(descriptives)-1]
    
    step_2_cleaning_descriptive_list = []
    
    for i in range(len(junk_3)):
        for j in range(len(descriptives)):
            if junk_3[i].find(descriptives[j])!=-1:
                step_2_cleaning_descriptive_list.append(junk_3[i])
                break
            else:
                continue
                
    junk_4 = np.setdiff1d(junk_3,step_2_cleaning_descriptive_list)
    
    step_3_cleaning_moods_list = []
    for i in range(len(junk_4)):
        for j in range(len(moods)):
            if jarowinkler.similarity(moods[j], junk_4[i]) > score_level :
                step_3_cleaning_moods_list.append(junk_4[i])
                
    step_3_cleaning_moods_list = np.unique(step_3_cleaning_moods_list)
    junk_5 = np.setdiff1d(junk_4,step_3_cleaning_moods_list)

    step_3_cleaning_genres_list = []

    for i in range(len(junk_5)):
        for j in range(len(genres)):
            if jarowinkler.similarity(genres[j], junk_5[i]) > score_level :
                step_3_cleaning_genres_list.append(junk_5[i])
    
    step_3_cleaning_genres_list = np.unique(step_3_cleaning_genres_list)            
    
    junk_6 = np.setdiff1d(junk_5, step_3_cleaning_genres_list)
    
    
    #compiling 
    list_1 = [(keys,w_l[keys]) for keys in w_l.keys() if w_l[keys] != 'junk']
    list_2 = [(x,'descriptive') for x in step_2_cleaning_descriptive_list] + [(x,'moods') for x in step_2_cleaning_moods_list] + [(x,'genres') for x in step_2_cleaning_genres_list]
    list_3 = [(x,'genres') for x in step_3_cleaning_genres_list] + [(x,'moods') for x in step_3_cleaning_moods_list] 
    list_4 = [(x,'junk') for x in junk_6]
    
    final_filtered_tags = list_1 + list_2 + list_3 + list_4
    
    return(final_filtered_tags)


# In[ ]:





# In[ ]:




