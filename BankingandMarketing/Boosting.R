# Use xgboost algorithm to make binary classification with random search for hypertuning parameters
#Load libraries
library(data.table)
library(caret)
library(xgboost)
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

# Random Search
ctrl <- makeTuneControlRandom(maxit = 15L)

library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

#stop parallelization
parallelStop()