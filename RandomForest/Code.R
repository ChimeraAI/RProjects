library(data.table)

#Read and separate raw data
rawData <- read.csv("bank-full.csv", sep = ";", header = TRUE)

#Convert to DataFrame
setDT(rawData)

#Set seed
set.seed(11)

#Shuffle Data
rawData <- rawData[sample(nrow(rawData)),]

#Split rawData into training and testing data
library(caTools)
tempData <- sample.split(rawData$y, SplitRatio = 0.7)
trainData <- rawData[tempData]
testData <- rawData[!tempData]

tr_target <- trainData$y
test_target <- testData$y

new_tr <- model.matrix(~.+0, data = trainData[,-c("y")])
new_test <- model.matrix(~.+0, data = testData[,-c("y")])

tr_target <- as.numeric(tr_target) - 1
test_target <- as.numeric(test_target) - 1

library(xgboost)
dTrain <- xgb.DMatrix(data = new_tr, label = tr_target)
dTest <- xgb.DMatrix(data = new_test, label = test_target)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)