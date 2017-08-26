# Implementation of xgboost
# Load libraries ---------------------------------------
library(caret)
library(caTools)
library(xgboost)
library(Matrix)
library(car)

dataFrame <- read.csv("bank-full.csv", sep = ";", header = TRUE)

dataFrame$day <- NULL

dataFrame$y <- as.numeric(dataFrame$y)

ohe_feats = c("job","marital","education","default","housing","loan","contact","poutcome","month")

dummies <- dummyVars(~ job + marital + education + default + housing + loan + contact + poutcome + month, data = dataFrame)

df_all_ohe <- data.frame(predict(dummies, newdata = dataFrame))

df_all_combined <- cbind(dataFrame[,-c(which(colnames(dataFrame) %in% ohe_feats))],df_all_ohe)

# Shuffle data
df_all_combined <- df_all_combined[sample(nrow(df_all_combined)),]

# Split data
tempData <- sample.split(df_all_combined$y, SplitRatio = 0.7)
trainData <- df_all_combined[tempData,]
testData <- df_all_combined[!tempData,]

drop <- ("y")

xgb <- xgboost(data = data.matrix(trainData[,!(names(trainData) %in% drop)]), 
               label = trainData$y, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 51,
               nthread = 3
)

y_pred <- predict(xgb, data.matrix(trainData[,!(names(trainData) %in% drop)]))








