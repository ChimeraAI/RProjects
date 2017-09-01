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

#Set seed
set.seed(11)

tsk <-  makeClassifTask(data = rawData, target = "y")

base = c("classif.rpart", "classif.lda", "classif.svm")
lrns = lapply(base, makeLearner)
lrns = lapply(lrns, setPredictType, "prob")
m = makeStackedLearner(base.learners = lrns,
                       predict.type = "prob", method = "hill.climb")
tmp = train(m, tsk)
res = predict(tmp, tsk)

library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())