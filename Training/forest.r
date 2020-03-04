library(tree)
library(MASS)
library(gbm)
library(dplyr)
library(randomForest)


# import data
rf.train <- write.csv("rf_train.csv", header =T)
rf.test <- write.csv("rf_test.csv",header = T)
feature <- read.csv("feature.csv",header = T)
train.size <- dim(rf.train)[1]

# select features what you think is reasonable
# the first column of feature is names of variable,
a <- cbind(select(rf.train, feature[1:38,1]),satisfied)

# given testing data, not needed when doing experiment
b <- select(rf.test, feature[1:38,1])

# split the training data into training and testing data
id <- sample(1:train.size,train.size*0.8)
a.new <- a[id,]

  
rf.fit.try = randomForest(satisfied~., 
                            data = a, 
                            ntree = 500, 
                            importance = TRUE)

# a.new now is the rest data which is testing data
a.new <- a[-id,]

# compute the accuracy
mean(predict(rf.fit.try,a.new) == a.new$satis)


# new importance ranking
imp.try <- varImp(rf.fit.try)
imp.order.try <- imp.try[order(imp.try[,1],decreasing = TRUE),]
imp.order.try

