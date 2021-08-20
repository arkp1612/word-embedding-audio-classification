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

         
[Unique tag list](http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt) contains the list of all unique tags that occur in the dataset. 

The filtered tag dictionery is attached in this repository in the `categorized_tags.txt` and can be replicated using the `tag_filtering.py` file in the EDA and Data Input Prep Folder.

There are 3 stages of filtering that we performed here and we were able to put 80% of the tags into categories. 20% junk tags out of a total of 522367 tags still contain some important music related tags like the artist names but they are left for now. The 80% filted tags check for the presence of emotions, descriptives, genres and numbers in the tags and classify into categories. Also, we use a typo correction algorithm which uses the Jarowinkler similarity criteria for identifying typos. If the Jrowinkler similarity score is greater than 0.9 only then we decide to classify the tag into a category. This value was selected on the basis of some experiments and we decided this made sense. This argument can be changed if you are trying to replicate the code using the "score_level" argument. 


We train the models for 50 tags - which are seacted to represent genres, decriptions, emotions and numeric tags. The selected 18 genre tags are - 'rock', 'pop', 'indie', 'electronic', 'dance', 'alternative rock', 'jazz', 'singer-songwriter', 'metal', 'chillout', 'classic rock', 'soul', 'indie rock', 'electronica', 'folk', 'instrumental', 'punk', 'oldies'; the 18 selected emotion tags are - 'mellow', 'sexy', 'loved', 'sad', 'happy', 'good', 'romantic', 'melancholic', 'great', 'dark', 'dreamy','hot', 'energetic', 'calm', 'funny', 'haunting', 'intense', 'alternative'; the 9 selected descriptive tags are - 'beautiful', 'awesome', 'british', 'chill', 'american', 'cool', 'favorite', 'acoustic', 'party' and finally the numeric tags are - '2000s', '80s', '90s', '60s', '70s'. This is also done to compare the performance of the word-embeddings concept for each ctaegory of the tags.


## Collapsing function - Stage II
The next step is to develop the collapser function. This function would essentially look at the list we prepared in the tag filtering stage and pick up the "sensible tags" that shoul be associated with tags which do not make sense. It is simply a cleaner function.

_Examples -_
1. _British rock metal -> rock metal_
2. _hiphop -> hip hop_
3. _80s metal -> metal_ etc.

This is essential to our tag cleaning process as we are reducing junk tags significantly and turning meaningless tags to meaningful tags which can be used for improving their representation in the dataset and hence improve model prediction. 

On running this python script given in `collapsed_tags_list_creator.py` you get the results stores in `sensible_tags_encoding.txt`. Note that these are created only for our selected top 50 tags, to avoid out of memeory error in model training.


## Collapsing function - Stage III
This collapsing function incorporates the word embeddings and creates a list of original and changed tags incorporating the word embedding logic. We looked at the distribution of the distances of the top 50 tags with each tag in the last.fm dataset. The distances for each tag are more or less distrubuted normally and we then use the percentile of the normal distribution with mean and standard deviation calculated for each of the selected tags. The tags from the dataset which are in the top 0.1%  neigbourhood of each selected tag is collapsed into them. Since we obtain vectors for each of the words, calcualting distance is easy - it is just the l2 norm between two vectors. So if the distance between two tags is less than 0.001 percentile of the normal distribution obtaied for each selected tag, they are collapsed into them. 


On running this python script given in `word_embedder.py` you get the results stores in `word_embedding_encoding.txt` in the EDA and Data Input Prep folder. Note that these are created only for our selected top 50 tags, to avoid out of memeory error in model training.


## The Models
The scripts for the models run are stored in the `Models` folder. To replicate any result, you just need to run the python script corresponding to the audio input format and the stage number which is defined in the model name. 

## Replicating results on the test datasets
The scripts for these are stored in the `Per Category Metrics Evaluation Code` folder. To replicate any result, just run the python script corresponding to the audio input format and stage number. 

## `get_embedded_tags.py`
This is a function we created to looked at the collapsed tags in the stage III of the collapsing functions. So if we want to look at what tags were collapsed into the tag - 'happy' we can simply run `python get_embedded_tags.py -- 'happy'` to get the results. The top 50 rows of the result of this code gives the follwoing output - `'happy', '128bitforbehappy', '2 happy or 2 fucked up','2011 happy beginnings male lead singers', '300happy fav','4-amhappy-dancing', '70s happy mood-lifter','90s happy monday ofh', ':happy cry:', 'a happy sort of sad','a sad sort of happy', 'a2c-happy', 'abc-many happy returns','acoustic makes me happy', 'air - lucky and unhappy', 'akes you happy', 'alan menken - happy end in agrabah','alegre happy froh', 'all kinds of happy','all we want is to be happy', 'almost happy', 'altered image-happy birthday', 'always happy to hear it','always make me happy', 'always makes me happy', 'amazing happy','angelique kidjo - happy xmas', 'are we happy yet','are you happy', 'avril - happy ending','avril lavigne - my happy ending', 'avril lavigne-my happy ending','bands with a lot of people make me happy', 'be happy', 'be happy dammit', 'be happy right now', 'being happy','being in swooshy love with a permanent smile and happy tear in eye', 'bellagrl happy', 'ben folds on happy pills', 'berlin happy', 'bert kaempfert - that happy feeling', 'bobby mc ferrin-dont worry be happy', 'brett dennen - happy music', 'bright and happy','brings happy thoughts', 'britney spears -- born to make you happy', 'britney spears- born to make u happy', 'btcmhappyhour', 'bubbly happy'`. We can see word embeddings work well in collapsing relavant tags to our selected tags. They however do not work so well for the numeric tags - casuing a lot of junk tags to be collapsed into them. 


## The Results
Figure below shows the training and validation loss across epochs for all model stages for the waveform and spectrogram models. We can see that the validation loss for the mel-spectrogram models is more unstable and we only select the model corresponding to the best epoch. Looking at the graphs, there is some evidence of over-fitting in the model for the waveform architectures, and we again only use the best epoch model specification for our selected model. 

<img width="343" alt="loss" src="https://user-images.githubusercontent.com/41041168/130158603-8cc7dfbc-fc80-4626-aff7-481e82c80f4c.png">

Figures below shows the training and validation ROC-AUC  per epoch for waveform and log-mel spectrogram models for all stages. There is again some evidence of over-fitting for the waveform models. We only select the model corresponding to the best epoch. For the stage I log-mel spectrogram model, the validation ROC-AUC seems to be stabilising after the 5th epoch. For stage II and stage III log-mel spectrogram models, there is some evidence that further training could have improved the model performance since the ROC-AUC is still increasing with each epoch. There is a similar behaviour for the PR-AUC metric for all models.



<p float="left">
  <img src="https://user-images.githubusercontent.com/41041168/130158780-432c5658-7fec-4ac2-b6db-a1b5940e5846.png" width="343" />
  <img src="https://user-images.githubusercontent.com/41041168/130158786-cf5a8530-a720-4f0b-8972-af8de11c803b.png" width="343" /> 
</p>


Table below show the model performance on the test set for waveform and log-mel spectrogram models for all three stages. We can see that the ROC-AUC metric for both waveform and log-mel spectrogram models is decreasing for stage II and stage III in comparison to stage I. But the PR-AUC is increasing for stage II and stage III for both waveform and log-mel spectrogram models.


|**Model Name** |**ROC-AUC**  |**PR-AUC**|
| ------------- | ------------- |------------- |
|Stage I Waveform (ours) | 84.39 | 29.41 |
|Stage II Waveform (ours) |82.07 | 30.11 |
|Stage III Waveform (ours) | 79.17 | 34.46 |
|Stage I Log-Mel Spectrogram (ours) | 85.05 | 31.38 |
|Stage II Log-Mel Spectrogram (ours) | 83.02 | 31.79 |
|Stage III Log-Mel Spectrogram (ours) | 80.25 | **35.84** |
|Waveform   (Pons, et al., 2018) | 87.41 | 28.53 |
|Log-mel-spectrogram   (Pons, et al., 2018) | **88.75** | 31.24|


Table below displays the tag category wise average evaluation metrics on the test dataset. We can observe a general decline in the ROC-AUC over the three stages. This can also be attributed to the unbalance in the dataset. And for unbalanced dataset, we know PR-AUC gives a  better evaluation of the model. The PR-AUC shows the maximum improvement for stage III in comparison to stage I for the emotion tags, followed by descriptive tags and then the genre tags. For the numeric tags, Stage II shows the best results. Simple cleaning of tags as shown in Stage II also shows a significant improvement in results in comparison to Stage I.

|   | ROC-AUC |ROC-AUC  | ROC-AUC| PR-AUC |PR-AUC  |PR-AUC   |
|---|---|---|---|---|---|---|
**Tag Category** | **Stage I** | **Stage II** | **Stage III** | **Stage I** | **Stage II** |**Stage III** |
|descriptive | 72.30 | 71.16 | 68.81 | 12.42 | 14.41 | 18.04 |
|emotion | 74.34 | 73.49 | 70.23 | 7.69 | 10.72 | 14.57 |
|genre | 85.03 | 83.32 | 81.73 | 35.85 | 36.38 | 40.50 |
|numeric | 81.40 | 80.23 | 78.81 | 20.80 | 22.48 | 20.74|


We have shown that incorporating semantics and adding information to poorly represented tags with similar context helps improve the model PR-AUC metric in a folkonomy. It is observed that the overall increase in the test PR-AUC from stage I to stage III is 17.17% for the waveform model and 14.21% for the log-mel spectrogram model. We know that the waveform model assumes no domain knowledge and Pons et al concluded that the log-mel spectrogram which incorporates the domain knowledge works better for audio classification. There is a higher increase in betterment in model performance for the waveform model so we can say that using word-semantics for this domain knowledge free model shows promise for this approach. The best model was still the log-mel spectrogram model but its performance using word-embeddings did improve in comparison to the baseline model. 



Figures below show the per tag ROC-AUC and PR-AUC on the test set, if the reader also wishes to examine these.


**per tag ROC-AUC metrics for log-mel spectrogram model**


![image](https://user-images.githubusercontent.com/41041168/130160385-9c50cce9-8f94-4106-8891-702abe2f4d33.png)


**per tag ROC-PR metrics for log-mel spectrogram model**


![image](https://user-images.githubusercontent.com/41041168/130160477-3f7d6552-be53-462a-a351-522d7471a301.png)


**per tag ROC-AUC metrics for waveform model**


![image](https://user-images.githubusercontent.com/41041168/130160549-1a856877-9766-4e8c-8414-92312f412830.png)


**per tag PR-AUC metrics for waveform model**


![image](https://user-images.githubusercontent.com/41041168/130160580-69466717-c858-4a11-99b4-cc5f1747b3af.png)
