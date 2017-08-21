library(RCurl)
library(caret)
urlfile <-'https://raw.githubusercontent.com/hadley/fueleconomy/master/data-raw/vehicles.csv'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
vehicles <- read.csv(textConnection(x))

vehicles <- vehicles[names(vehicles)[1:24]]
vehicles <- data.frame(lapply(vehicles, as.character), stringsAsFactors=FALSE)
vehicles <- data.frame(lapply(vehicles, as.numeric))
vehicles[is.na(vehicles)] <- 0
vehicles$cylinders <- ifelse(vehicles$cylinders == 6, 1,0)

set.seed(1234)
vehicles <- vehicles[sample(nrow(vehicles)),]
split <- floor(nrow(vehicles)/3)
ensembleData <- vehicles[0:split,]
blenderData <- vehicles[(split+1):(split*2),]
testingData <- vehicles[(split*2+1):nrow(vehicles),]

labelName <- 'cylinders'
predictors <- names(ensembleData)[names(ensembleData) != labelName]

myControl <- trainControl(method='cv', number=3, returnResamp='none')

model_gbm <- train(ensembleData[,predictors], ensembleData[,labelName], method='gbm', trControl=myControl)

model_rpart <- train(ensembleData[,predictors], ensembleData[,labelName], method='rpart', trControl=myControl)

model_treebag <- train(ensembleData[,predictors], ensembleData[,labelName], method='treebag', trControl=myControl)