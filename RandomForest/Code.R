

set.seed(22)

dataFrame <- read.csv("bank-full.csv", sep = ";", header = TRUE)

apply(dataFrame,2,function(x) length(unique(x)))

dataFrame$day <- NULL

cols <- c("job","marital","education","default","housing","loan","contact","poutcome","y")
for(i in cols) {
  dataFrame[,i] = as.factor(dataFrame[,i])
}

barplot(table(dataFrame$education))
