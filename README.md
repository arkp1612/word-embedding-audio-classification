# Audio Classification using word embeddings
Github repo for my Masters Thesis

For the purpose of this project, we use the million song dataset. The idea is to use word embeddings for poorly represented tags in the dataset and improve model performance of audio classification. The various steps carried out for the project are discuessed below.


Since the million song database is derived from a folksonomy, the tags given out by users do not always make sense. Thus it is an important first step to study the nature of tags and remove any unnecessary tags which will lead to faulty working of the model in the later stages. Most models which have used the million song dataset so far have used the top 50 or 100 or more tags but not all of them. The first step that we have considered here is to filter these tags into broader categories, so we can also compare the overall model performance under different types of tags. This is discuessed in the next section. 


## Tag filtering 
We first look at the types of tags and notice that there are some broad categories of tags and we first create a dictionary to classify these tags. We hope that this would easily help us pick tag category and check which category works the best with word embeddings. Our bet is that the emotion tags would work best for the word embeddings. The tag categories identified were - 

1. Emotion tags (happy, sad, angry etc.)
2. Decriptive tags (American, cringy, awesome etc.)
3. Genre tags (rock, pop, alternative etc.)
4. Number tags (Top 5, 80s etc.) 
5. Junk tags (favorites, love, zzzzzzzz, etc.)

         
[Unique tag list](http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt) contains the list of all unique tags that occur in the dataset. The filtered tag dictionery is attached in this repository and the code used to do that is in the tag_filtering.py file.


There are 3 stages of filtering that we performed here and we were able to put 80% of the tags into categories. 20% junk tags out of a total of 522367 tags still contain some important music related tags like the artist names but they are left for now. The 80% filted tags check for the presence of emotions, descriptives, genres and numbers in the tags and classify into categories. Also, we use a typo correction algorithm which uses the Jarowinkler similarity criteria for identifying typos. If the Jrowinkler similarity score is greater than 0.9 only then we decide to classify the tag into a category. This value was selected on the basis of some experiments and we decided this made sense. This argument can be changed if you are trying to replicate the code using the "score_level" argument. 


## Collapsing function
The next step is to develop the collapser function. This function would essentially look at the list we prepared in the tag filtering stage and pick up the "sensible tags" that shoul be associated with tags which do not make sense. It is simply a cleaner function.

_Examples -_
1. _British rock metal -> rock metal_
2. _hiphop -> hip hop_
3. _80s metal -> metal_ etc.

This is essential to our tag cleaning process as we are reducing junk tags significantly and turning meaningless tags to meaningful tags which can be used for improving their representation in the dataset and hence improve model prediction. 
