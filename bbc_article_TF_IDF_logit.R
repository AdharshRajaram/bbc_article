library(tm)
library(NLP)
library(SnowballC) 
library(wordcloud)
library(RColorBrewer)
######Read the CSV file#########
bbc_dat<-read.table("bbc_dat.csv",
                    header=T,
                    sep = ",",
                    stringsAsFactors = F,
                    quote="\"",
                    comment.char = "",
                    strip.white = T
)


######Sort the data Randomly########
set.seed(1)
rand<-sample(nrow(bbc_dat))
bbc_dat<-bbc_dat[rand,]
row.names(bbc_dat)<-NULL

########Build a Corpus##########
bbc_corpus<-Corpus(VectorSource(bbc_dat$text))

#display documents of Corpus
#writeLines(as.character(bbc_corpus[[1]]))

########Preprocess/Clean Corpus########
bbc_corpus<-tm_map(bbc_corpus,tolower)
bbc_corpus<-tm_map(bbc_corpus,stripWhitespace)
bbc_corpus<-tm_map(bbc_corpus,removePunctuation)
bbc_corpus<-tm_map(bbc_corpus,removeNumbers)

########Remove StopWords##########
bbc_corpus<-tm_map(bbc_corpus,removeWords,stopwords('english'))

#######Stem Words############
bbc_corpus<-tm_map(bbc_corpus,PlainTextDocument)
bbc_corpus<-tm_map(bbc_corpus,stemDocument)


#inspect the entire Corpus after preprocessing
#writeLines(as.character(bbc_corpus),con="bbc_corpus.txt")

##########DocumentTermMatrix#########
bbc_dtm<-DocumentTermMatrix(bbc_corpus)


#########DTM explore############
index1<-which(bbc_dat$category=="sport")
index2<-which(bbc_dat$category=="business")
index3<-which(bbc_dat$category=="politics")
index4<-which(bbc_dat$category=="tech")
index5<-which(bbc_dat$category=="entertainment")
wordF<-data.frame(Total=apply(bbc_dtm_sparse, 2, sum))
wordF<-cbind(wordF,data.frame(Sport=apply(bbc_dtm_sparse[index1,], 2, sum)))
wordF<-cbind(wordF,data.frame(Business=apply(bbc_dtm_sparse[index2,], 2, sum)))
wordF<-cbind(wordF,data.frame(Politics=apply(bbc_dtm_sparse[index3,], 2, sum)))
wordF<-cbind(wordF,data.frame(Tech=apply(bbc_dtm_sparse[index4,], 2, sum)))
wordF<-cbind(wordF,data.frame(Entertainment=apply(bbc_dtm_sparse[index5,], 2, sum)))
#write.csv(wordF,"frequency_sparse.csv")
#findFreqTerms(bbc_dtm[index1,],150)

#findAssocs(bbc_dtm,"said",0.6)
#wordcloud for each category
wordcloud(bbc_corpus[index1],min.freq=150,max.words=100,fixed.asp = FALSE,rot.per = 0)
wordcloud(bbc_corpus[index2],min.freq=150,max.words=100,fixed.asp = FALSE,rot.per = 0)
wordcloud(bbc_corpus[index3],min.freq=150,max.words=100,fixed.asp = FALSE,rot.per = 0)
wordcloud(bbc_corpus[index4],min.freq=150,max.words=100,fixed.asp = FALSE,rot.per = 0)
wordcloud(bbc_corpus[index5],min.freq=150,max.words=100,fixed.asp = FALSE,rot.per = 0)


######TF-IDF weight##########
bbc_dtm_sparse<-removeSparseTerms(bbc_dtm,0.96)
bbc_dtm_sparse<-weightTfIdf(bbc_dtm_sparse)
bbc_dtm_mat<-as.matrix(bbc_dtm_sparse)


