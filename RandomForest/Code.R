# Load libraries ---------------------------------------
library(magrittr)
library(mlbench)
library(caret)
library(caretEnsemble)

set.seed(22)

dataFrame <- read.csv("bank-full.csv", sep = ";", header = TRUE)

apply(dataFrame,2,function(x) length(unique(x)))


# Getting rid of irrelevant data ----------------------------------------
dataFrame$day <- NULL
# Not sure the purpose of this feature. Don't want to deal with unknow features
dataFrame$balance <- NULL


# Converting categorical features to factor variables as oppose to numerical -----------------------------------
cols <- c("job","marital","education","default","housing","loan","contact","poutcome","y")
for(i in cols) {
  dataFrame[,i] = as.factor(dataFrame[,i])
}

# Example of Stacking algorithms
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)





