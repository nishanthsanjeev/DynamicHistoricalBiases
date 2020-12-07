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

# Steps for reproducing EngAll analyses:

## 1) Download pre-trained word vectors from HistWords:
Click [here](http://snap.stanford.edu/historical_embeddings/eng-all.zip) to download. (Warning: ~8.8 GB download size)

## 2) Generate .txt word vector files:

Run the file titled *'readpklandnpy.R'*. Make sure it is in the same directory as the file titled *'read_pickle.py'*.

## 3) Generate .DB (database) file to store word vectors:

Run the file titled *'import_histvecs.py'*.

## 4) Reproduce results:

Go to the *eng-all* directory, and run the .py files present in each folder (Frequency, loocv, and weat) to reproduce results so far. Make sure to change directories where applicable. Their equivalent baseline folders (i.e., Frequency_baseline, etc) may also be found, and the user can reproduce these as a baseline test.

Frequency files:

*average_freq_all_decades.py* - Measures how frequently the words that compose a category occur over all decades. Open file for more information. The plot file that is generated however, measures frequency *per decade* instead. (Need to make changes) <br />

*ind_freq_per_decade.py* - Find the individual frequency of each word in a stigma category, per decade.<br />

LOOCV:

*loocv.py* - Computes a list of WEAT effect sizes, as a cross-validation technique. (Leave-one-out-cross-validation)<br />

WEAT:

*avg_weat_all_decades.py* - Computes avg WEAT effect size across all decades.<br />
*weat_decade.py* - Computes WEAT effect size per decade for each stigma category.<br />



## Additional information:

Note: When running the Python frequency scripts - make sure to run using Python2.<br />
All other .py files - Python3 works just fine.
