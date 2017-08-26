library(RCurl)
library(data.table)
library(xgboost)

train.url <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
test.url <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')

setcol <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race",
            "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target")

trainData <- read.table(textConnection(train.url), header = F, sep = ",", col.names = setcol, na.strings = c(" ?"), stringsAsFactors = F)
testData <- read.table(textConnection(test.url), header = F,sep = ",", col.names = setcol,skip = 1, na.strings = c(" ?"), stringsAsFactors = F)

setDT(trainData)
setDT(testData)

table(is.na(trainData))
sapply(trainData, function(x) sum(is.na(x))/length(x))*100

table(is.na(testData))
sapply(testData, function(x) sum(is.na(x))/length(x))*100

library(stringr)
testData[,target <- substr(target, start = 1, stop = nchar(target) - 1)]

char_col <- colnames(trainData)[ sapply (testData,is.character)]
for(i in char_col) set(trainData,j=i,value = str_trim(trainData[[i]],side = "left"))

for(i in char_col) set(testData,j=i,value = str_trim(testData[[i]],side = "left"))

trainData[is.na(trainData)] <- "Missing"
testData[is.na(testData)] <- "Missing"

labels <- trainData$target
ts_labels <- testData$target
new_tr <- model.matrix(~.+0,data = trainDataata[,-c("target"),with=F])
new_ts <- model.matrix(~.+0,data = testData[,-c("target"),with=F])

labels <- as.numeric(as.factor(labels))-1
ts_labels <- as.numeric(as.factor(ts_labels))-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels)

dtest <- xgb.DMatrix(data = new_ts,label = ts_labels)

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

min(xgbcv$test.error.mean)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 79, watchlist = list(val=dtest,train=dtrain), print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")

xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

library(caret)
confusionMatrix (xgbpred, ts_label)

mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 

library(mlr)

fact_col <- colnames(trainData)[sapply(trainData,is.character)]

for(i in fact_col) set(trainData,j=i,value = factor(trainData[[i]]))
for(i in fact_col) set(testData,j=i,value = factor(testData[[i]]))

traintask <- makeClassifTask (data = trainData,target = "target")
testtask <- makeClassifTask (data = testData,target = "target")

traintask <- createDummyFeatures (obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)

lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

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
  