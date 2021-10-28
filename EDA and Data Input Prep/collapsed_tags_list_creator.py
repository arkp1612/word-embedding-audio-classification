#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def collapsed_tags_list_creator():
        
    import re, numpy as np, requests,json
    req = urllib.request.Request('https://gist.githubusercontent.com/sampsyo/1241307/raw/208ab2e4b5b576ebc51d801b039f93ee2bbc33ea/genres.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    
    genres = the_page.split(b'\n')
    
    genres = [x.lower() for x in list(map(decoder,genres))]
    
    resp =  urllib.request.urlopen('https://musicbrainz.org/genres')
    html_bytes = resp.read()
    html = html_bytes.decode("utf-8")
    gen =  re.split('<bdi>|</bdi>', html[9067:78729])[1::2]
    
    genres = gen + genres
    genres = genres[:-1]
    genres = np.unique(genres)
    
    
    req = urllib.request.Request('https://raw.githubusercontent.com/taikuukaits/SimpleWordlists/master/Wordlist-Adjectives-All.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    
    descriptives = the_page.split(b'\n')
    descriptives = [x.lower() for x in list(map(decoder,descriptives))]
    descriptives = descriptives[:-1]
    
    url = 'https://raw.githubusercontent.com/dariusk/corpora/master/data/humans/moods.json'
    resp = requests.get(url)
    data = json.loads(resp.text)
    moods = data['moods']
    moods = moods[:-1]
    
    import urllib
    req = urllib.request.Request('http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()

    words = the_page.replace(b'\t',b'\n').split(b'\n')

    freq = words[1:len(words):2]
    words = words[0:(len(words)+1):2]

    all_tags = [words[i].decode('utf-8').lower() for i in range(len(words))]

        
    def decoder(x):
        return x.decode('utf-8')
    
    req = urllib.request.Request('https://raw.githubusercontent.com/arkp1612/word-embedding-audio-classification/main/categorized_tags.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()
    
    tags = the_page.split(b'\n')
    tags = [x.lower() for x in list(map(decoder,tags))]
    tags = [x.replace('\r','').split('\t') for x in tags]
    tags = tags[:-1]
    
    genre_tags = [x[0] for x in tags if x[1]=='genres']
    mood_tags = [x[0] for x in tags if x[1]=='moods']
    descriptive_tags = [x[0] for x in tags if x[1]=='descriptive']
    number_tags = [x[0] for x in tags if x[1]=='number']
    junk_tags = [x[0] for x in tags if x[1]=='junk']

    
    def sensible_tag(list_1,word,score_level):
        from strsimpy.jaro_winkler import JaroWinkler
        jarowinkler = JaroWinkler()
        for x in list_1:
            if x == word:
                return(x)
        
        for x in list_1:
            if x in word:
                return(x)
    
        for x in list_1:
            if jarowinkler.similarity(x,word) >= score_level:
                return(x)  
            
    sensible_genre_tags = [[x,sensible_tag(genres,x,0.9)] for x in genre_tags]
    sensible_moods_tags = [[x,sensible_tag(moods,x,0.9)] for x in mood_tags]
    sensible_descriptive_tags = [[x,sensible_tag(descriptives,x,0.9)] for x in descriptive_tags]
    all_sensible_tags = sensible_genre_tags + sensible_moods_tags + sensible_descriptive_tags + [[x,x] for x in number_tags] + [[x,x] for x in junk_tags]
    
    with open("sensible_tags_encoding.txt", 'w',encoding="utf-8") as output:
    for row in all_sensible_tags:
        output.write(str(row[0]) + '\t' + str(row[1]) + '\n')


# In[ ]:




