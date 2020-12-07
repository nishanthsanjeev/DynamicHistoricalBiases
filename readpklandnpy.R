# load packages
require(reticulate)
require(lsa)


# set up R for integration with python
py_install("pandas")
py_install("pickle")
source_python("read_pickle.py")
np <- import("numpy")



##READ IN COHA EMBEDDINGS ----
## Reading in pkl and ny pretrained COHA embeddings for use in R 
# (and unpickling them for use otherwise as .txt files)


# read in the words, wordvecs, and merge into a dataframe
##Yoav - change directory below to location that has the .npy and .pkl embedding files
#setwd("/media/nishanthsanjeev/2CC66AE5C66AAF30/Embeddings/eng-all/sgns")

decades <- seq(1800, 1810, by = 10)
pklfiles <- paste(decades, "-vocab.pkl", sep = "")
wfiles <- paste(decades, "-w.npy", sep = "")
pkls <- list()
wordvecs <- list()
wordvecs.dat <- list()

for (i in 1:length(decades)){
  pkls[[i]] <- read_pickle_file(pklfiles[i])
  wordvecs[[i]] <- np$load(wfiles[i])
  wordvecs.dat[[i]] <- as.data.frame(wordvecs[[i]])
  rownames(wordvecs.dat[[i]]) <- pkls[[i]]
}


## save out each decade as individual text files
for (i in 1:length(wordvecs.dat)) {
  write.table(wordvecs.dat[[i]], file = paste("vectors", decades[i], ".txt", sep = ""),
            row.names = TRUE, col.names = FALSE, quote = FALSE)
}