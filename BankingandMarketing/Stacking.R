#Load libraries
library(data.table)
library(caret)
library(xgboost)
library(mlr)
library(caTools)
library(parallel)
library(parallelMap)

#Read and separate raw data
rawData <- read.csv("BankingandMarketing/bank-full.csv", sep = ";", header = TRUE)

#Convert to DataFrame
as.data.frame(rawData) 

fact_col <- colnames(rawData)[sapply(rawData,is.character)]

for(i in fact_col) set(rawData,j=i,value = factor(rawData[[i]]))

#Set seed
set.seed(11)

tsk <-  makeClassifTask(data = rawData, target = "y")

tsk <- createDummyFeatures (obj = tsk)

base = c("classif.xgboost", "classif.randomForest")
lrns = lapply(base, makeLearner)
lrns = lapply(lrns, setPredictType, "prob")
m = makeStackedLearner(base.learners = lrns,
                       predict.type = "prob", method = "hill.climb")

parallelStartSocket(cpus = detectCores())

tmp = train(m, tsk)
res = predict(tmp, tsk)

parallelStop()

performance(res, measures = acc)
