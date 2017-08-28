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


# Setting parameters for xgboost
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dTrain, 
                 nrounds = 100, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

#first default - model training
xgb1 <- xgb.train (params = params, data = dTrain, nrounds = 79, 
                   watchlist = list(val=dTest,train=dTrain), print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dTest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#confusion matrix
library(caret)
confusionMatrix (xgbpred, test_target)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 

fact_col <- colnames(trainData)[sapply(trainData,is.character)]

for(i in fact_col) set(trainData,j=i,value = factor(trainData[[i]]))
for (i in fact_col) set(testData,j=i,value = factor(testData[[i]]))

library(mlr)
traintask <- makeClassifTask (data = trainData,target = "y")
testtask <- makeClassifTask (data = testData,target = "y")

traintask <- createDummyFeatures (obj = traintask)
testtask <- createDummyFeatures (obj = testtask)

lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

ctrl <- makeTuneControlRandom(maxit = 10L)

library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgmodel <- train(learner = lrn_tune,task = traintask)

xgpred <- predict(xgmodel,testtask)

confusionMatrix(xgpred$data$response,xgpred$data$truth)





