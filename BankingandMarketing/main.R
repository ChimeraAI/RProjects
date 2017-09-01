#Load libraries
library(data.table)
library(caret)
library(xgboost)
library(mlr)
library(caTools)
library(parallel)
library(parallelMap)

# Source functions from other files
source("Stacking.R")
source("Boosting.R")
source("Bagging.R")

#Read and separate raw data
rawData <- read.csv("BankingandMarketing/bank-full.csv", sep = ";", header = TRUE)

#Convert to DataFrame dsaf
setDT(rawData)

#Set seed
set.seed(11)

