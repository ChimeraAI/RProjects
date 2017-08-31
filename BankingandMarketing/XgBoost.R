# Use xgboost algorithm to make binary classification with grid search for hypertuning parameters
#Load libraries
library(data.table)
library(caret)
library(xgboost)
# Has the grid search option for hypertuning as oppose to xgboost library
library(mlr)
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

#Convert characters to factors
fact_col <- colnames(trainData)[sapply(trainData,is.character)]

for(i in fact_col) set(trainData,j=i,value = factor(trainData[[i]]))
for (i in fact_col) set(testData,j=i,value = factor(testData[[i]]))

#Create Tasks
traintask <- makeClassifTask (data = trainData,target = "y")
testtask <- makeClassifTask (data = testData,target = "y")

# Perform one hot encoding
traintask <- createDummyFeatures (obj = traintask)
testtask <- createDummyFeatures (obj = testtask)

#Create Learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")

lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, colsample_bytree = 1, 
                      booster = gbtree, subsample = 1, min_child_weight = 1, max_depth = 7)

params <- makeParamSet( makeNumericParam("eta",lower = 0.05L, upper = 0.15L),
                        makeIntegerParam("gamma",lower = 0, upper = 6),
                        makeNumericParam("alpha",lower = 0.5,upper = 2))

# 10-fold cross validation
rdesc <- makeResampleDesc("CV",stratify = T,iters=10L)

# Grid search
ctrl <- makeTuneControlGrid(resolution = 15)

library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgmodel <- train(learner = lrn_tune,task = traintask)

xgpred <- predict(xgmodel,testtask)

confusionMatrix(xgpred$data$response,xgpred$data$truth) 