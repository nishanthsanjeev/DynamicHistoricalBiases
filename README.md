# Dynamic Historical Biases

A collection of methods to analyze the changes in social biases and stereotypes over the past 200 years, using word embedding techniques on historical corpora.

## For the historical sources: 

We utilize **HistWords**, a brilliant collection of pre-trained vectors available at: https://nlp.stanford.edu/projects/histwords/
(Hamilton et al, 2018)

Here, we provide the tools to conduct analyses on the following corpora:

Google N-grams eng-all (EngAll) <br />
Google N-grams eng-fic (EngFic) <br />
*Corpus of Historical American English* (COHA)

## Requirements:
R <br />
Python2 <br />
Python3

# Steps for reproducing eng-all analyses:
## 1) Download pre-trained word vectors from HistWords:
Click [here](http://snap.stanford.edu/historical_embeddings/eng-all.zip) to download. (Warning: ~8.8 GB in size)

## 2) Generate .txt word vector files:

Run the file titled 'readpklandnpy.R'. Make sure it is in the same directory as the file titled 'read_pickle.py'.
<br />

## Information on each file:

Note: When running the *freqscript.py* file - make sure to run using Python2!<br />
All other .py files - Python3 works just fine.
