# Use randomForest algorithm to make binary classification with random search for hypertuning parameters
#Load libraries
library(data.table)
library(caret)
library(xgboost)
library(mlr)
library(caTools)

#Read and separate raw data
rawData <- read.csv("BankingandMarketing/bank-full.csv", sep = ";", header = TRUE)

#Convert to DataFrame
setDT(rawData)

#Check for missing data
sapply(rawData, function(x) sum(is.na(x))/length(x))

#Set seed
set.seed(11)

#Split rawData into training and testing data
tempData <- sample.split(rawData$y, SplitRatio = 0.7)
trainData <- rawData[tempData]
testData <- rawData[!tempData]

#Check for data imbalance between train and test data
setDT(trainData)[,.N/nrow(trainData),y]
setDT(testData)[,.N/nrow(testData),y]

#Convert characters to factors
fact_col <- colnames(trainData)[sapply(trainData,is.character)]

for(i in fact_col) set(trainData,j=i,value = factor(trainData[[i]]))
for (i in fact_col) set(testData,j=i,value = factor(testData[[i]]))

# Perform one hot encoding
traintask <- createDummyFeatures (obj = traintask)
testtask <- createDummyFeatures (obj = testtask)

#Create Tasks
traintask <- makeClassifTask (data = trainData,target = "y")
testtask <- makeClassifTask (data = testData,target = "y")

rf.lrn <- makeLearner("classif.randomForest")

#Random Forest Tuning
getParamSet(rf.lrn)
rf.lrn$par.vals <- list(ntree = 100L,
                        importance=TRUE)

#set parameter space
params <- makeParamSet(
  makeIntegerParam("mtry",lower = 2,upper = 10),
  makeIntegerParam("nodesize",lower = 10,upper = 50)
)

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=10L)

#Random Search
ctrl <- makeTuneControlRandom(maxit = 10L)

library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())

tune <- tuneParams(learner = rf.lrn
                   ,task = traintask
                   ,resampling = rdesc
                   ,measures = list(acc)
                   ,par.set = params
                   ,control = ctrl
                   ,show.info = T)

#stop parallelization
parallelStop()



