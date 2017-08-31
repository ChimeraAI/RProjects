# Load Libraries

library(data.table)
library(caret)
library(xgboost)
library(caTools)

#Read and separate raw data
rawData <- read.csv("bank-full.csv", sep = ";", header = TRUE)

#Convert to DataFrame
setDT(rawData)

#Set seed
set.seed(11)

#Split rawData into training and testing data
tempData <- sample.split(rawData$y, SplitRatio = 0.7)
trainData <- rawData[tempData]
testData <- rawData[!tempData]

tr_target <- trainData$y
test_target <- testData$y

new_tr <- model.matrix(~.+0, data = trainData[,-c("y")])
new_test <- model.matrix(~.+0, data = testData[,-c("y")])

tr_target <- as.numeric(tr_target) - 1
test_target <- as.numeric(test_target) - 1

dTrain <- xgb.DMatrix(data = new_tr, label = tr_target)
dTest <- xgb.DMatrix(data = new_test, label = test_target)


#----------------------------------------------------------------------------------

# Setting parameters for xgboost
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.07, gamma=5, max_depth=7, min_child_weight=1, subsample=1, colsample_bytree=1, alpha = 1)

xgbcv <- xgb.cv( params = params, data = dTrain, 
                 nrounds = 100, nfold = 10, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

xgb

#first default - model training
xgb1 <- xgb.train (params = params, data = dTrain, nrounds = 79, 
                   watchlist = list(val=dTest,train=dTrain), print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dTest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#-----------------------------------------------------------------------------------

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 

#confusion matrix
confusionMatrix (xgbpred, test_target)





