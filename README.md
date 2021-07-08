# Audio Classification using word embeddings
Github repo for my Masters Thesis

For the purpose of this project, we use the million song dataset. The idea is to use word embeddings for poorly represented tags in the dataset and improve model performance of audio classification. The various steps carried out for the project are discuessed below.


## Tag filtering 
We first look at the types of tags and notice that there are some broad categories of tags and we first create a dictionary to classify these tags. We hope that this would easily help us pick tag category and check which category works the best with word embeddings. Our bet is that the emotion tags would work best for the word embeddings. The tag categories identified were - 

1. Emotion tags (happy, sad, angry etc.)
2. Decriptive tags (American, cringy, awesome etc.)
3. Genre tags (rock, pop, alternative etc.)
4. Junk tags (favorites, love, zzzzzzzz, etc.)
         
[Unique tag list](http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt) contains the list of all unique tags that occur in the dataset. The filtered tag dictionery is attached in this repository and the code used to do that is in the tag_filtering.py file.
