import urllib
import numpy as np
import pandas as pd
import re

def collapsed_embedding(tag):
    req = urllib.request.Request('https://raw.githubusercontent.com/arkp1612/word-embedding-audio-classification/main/word_embedding_encoding.txt')
    with urllib.request.urlopen(req) as response:
        the_page = response.read()

    s3_words = the_page.replace(b'\t',b'\n').split(b'\n')


    os3_tag = s3_words[1:len(s3_words):2]
    os3_tag = [x.decode('utf-8') for x in os3_tag]

    s3_words = s3_words[0:(len(s3_words)+1):2]
    s3_words = [x.decode('utf-8') for x in s3_words]

    print([s3_words[i] for i in range(len(os3_tag)) if os3_tag[i]==tag])