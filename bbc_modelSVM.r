library(tm)
library(NLP)
library(SnowballC) 
library(nnet)
library(e1071)
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

##########DocumentTermMatrix#########
bbc_dtm<-DocumentTermMatrix(bbc_corpus)


######TF-IDF weight##########
bbc_dtm_mat<-removeSparseTerms(bbc_dtm,0.96)
bbc_dtm_mat<-weightTfIdf(bbc_dtm_mat)
bbc_dtm_mat<-as.data.frame(as.matrix(bbc_dtm_mat))
bbc_dtm_mat<-cbind(category=bbc_dat$category,bbc_dtm_mat)

#######Train & Validation Split#####
set.seed(100)
index<-sample(1:nrow(bbc_dtm_mat),0.70*nrow(bbc_dtm_mat),replace = FALSE)
bbc_trainig<-bbc_dtm_mat[index,]
bbc_valid<-bbc_dtm_mat[-index,]
row.names(bbc_trainig)<-NULL
row.names(bbc_valid)<-NULL

#############################Modelling########################################
model<-data.frame(Model="SVM",Iteration=1,Remark="First SVM Model",TError=0,AIC=0,RDeviance=0)
i<-1L                                 #Iteration number

######Run SVM:Model 1######
svm_model<-svm(bbc_trainig$category~.,data = bbc_trainig[,-1])
head(fitted(svm_model))
predicted<-data.frame(predict(svm_model,bbc_trainig[,-c(1)],type = "probs"))
model$TError[i]<-sum(1-(predicted[,1]==bbc_trainig$category))


#####Validate Model###########
predicted<-data.frame(predict(svm_model,bbc_valid[,-c(1)],type = "probs"))
model$VError[i]<-sum(1-(predicted[,1]==bbc_valid$category))
table(Actual=bbc_valid$category,Predicted=predicted[,1])


###################################Model2#########################################

######New Document Matrix###########
bbc_dtm<-DocumentTermMatrix(bbc_corpus)
#Remove Sparse terms which occur only once or twice
bbc_dtm_mat<-removeSparseTerms(bbc_dtm,0.99)

#########DTM explore############
index1<-which(bbc_dat$category=="sport")
index2<-which(bbc_dat$category=="business")
index3<-which(bbc_dat$category=="politics")
index4<-which(bbc_dat$category=="tech")
index5<-which(bbc_dat$category=="entertainment")
wordF<-data.frame(Total=apply(bbc_dtm_mat, 2, sum))
wordF<-cbind(wordF,data.frame(Sport=apply(bbc_dtm_mat[index1,], 2, sum)))
wordF<-cbind(wordF,data.frame(Business=apply(bbc_dtm_mat[index2,], 2, sum)))
wordF<-cbind(wordF,data.frame(Politics=apply(bbc_dtm_mat[index3,], 2, sum)))
wordF<-cbind(wordF,data.frame(Tech=apply(bbc_dtm_mat[index4,], 2, sum)))
wordF<-cbind(wordF,data.frame(Entertainment=apply(bbc_dtm_mat[index5,], 2, sum)))


#########Chossing only Important Features\Terms##########
observed_count<-wordF[,c(2:6)]
sport_c<-sum(bbc_dat$category=="sport")
business_c<-sum(bbc_dat$category=="business")
politics_c<-sum(bbc_dat$category=="politics")
tech_c<-sum(bbc_dat$category=="tech")
entertainment_c<-sum(bbc_dat$category=="entertainment")
Total_c<-nrow(bbc_dat)
expected_count_prob<-data.frame(row.names=row.names(observed_count))
expected_count_prob$Sport<-sport_c/Total_c
expected_count_prob$Business<-business_c/Total_c
expected_count_prob$Politics<-politics_c/Total_c
expected_count_prob$Tech<-tech_c/Total_c
expected_count_prob$Entertainment<-entertainment_c/Total_c
row.names(observed_count)<-NULL
row.names(expected_count_prob)<-NULL

p_val<-rep(0,nrow(observed_count))
for(i in 2:nrow(observed_count)){
  chi<-chisq.test(x=as.numeric(observed_count[i,]),p=as.numeric(expected_count_prob[i,]))
  p_val[i]<-chi$p.value
}
index<-which(p_val>0.05)
bbc_dtm_mat2<-bbc_dtm_mat[,-index]


######TF-IDF weight##########
bbc_dtm_mat2<-weightTfIdf(bbc_dtm_mat2)
bbc_dtm_mat2<-as.data.frame(as.matrix(bbc_dtm_mat2))
row.names(bbc_dtm_mat2)<-NULL
bbc_dtm_mat2<-cbind(category=bbc_dat$category,bbc_dtm_mat2)

#######Train & Validation Split#####
set.seed(100)
index<-sample(1:nrow(bbc_dtm_mat2),0.75*nrow(bbc_dtm_mat2),replace = FALSE)
bbc_trainig<-bbc_dtm_mat2[index,]
bbc_valid<-bbc_dtm_mat2[-index,]
row.names(bbc_trainig)<-NULL
row.names(bbc_valid)<-NULL


######Run SVM:Model 2######
i<-2L                                 
model[i,]<-c("svm_2",i,"Train:75,Valid:25,P_Value:0.05,sparse:0.99",0,0,0,0)

svm_model<-svm(bbc_trainig$category~.,data = bbc_trainig[,-1])
head(fitted(svm_model))
predicted<-data.frame(predict(svm_model,bbc_trainig[,-c(1)],type = "probs"))
model$TError[i]<-sum(1-(predicted[,1]==bbc_trainig$category))


#####Validate Model###########
predicted<-data.frame(predict(svm_model,bbc_valid[,-c(1)],type = "probs"))
model$VError[i]<-sum(1-(predicted[,1]==bbc_valid$category))
table(Actual=bbc_valid$category,Predicted=predicted[,1])