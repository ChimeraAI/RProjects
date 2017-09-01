
stackingFunc <- function(rawData) {

  #Convert to DataFrame
  as.data.frame(rawData) 
  
  # Convert characters to factors
  fact_col <- colnames(rawData)[sapply(rawData,is.character)]
  
  for(i in fact_col) set(rawData,j=i,value = factor(rawData[[i]]))
  
  # Create Task
  tsk <-  makeClassifTask(data = rawData, target = "y")
  
  # Perform one hot encoding
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
  
  return (performance(res, measures = acc))



}