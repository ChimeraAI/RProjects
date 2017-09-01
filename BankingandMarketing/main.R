#Load libraries
library(data.table)
library(caret)
library(xgboost)
library(mlr)
library(caTools)
library(parallel)
library(parallelMap)

# Source functions from other files
source("BankingandMarketing/Stacking.R")
source("BankingandMarketing/Boosting.R")
source("BankingandMarketing/Bagging.R")

#Read and separate raw data
rawData <- read.csv("BankingandMarketing/bank-full.csv", sep = ";", header = TRUE)

#Convert to DataFrame dsaf
setDT(rawData)

#Set seed
set.seed(11)

#Boosting
boostingFunc(rawData)

#Stacking
#stackingFunc(rawData)

#Bagging
#baggingFunc(rawData)



