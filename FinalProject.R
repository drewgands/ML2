rm(list=ls())
library(tidyverse)
library(ggplot2)
library(corrplot)
library(gbm)
library(xgboost)
library(randomForest)
library(glmnet)
options(scipen=999)

###Importing Data
properties_2016 <- read.csv("properties_2016.csv",stringsAsFactors = TRUE) #X
train_2016 <- read.csv("train_2016_v2.csv",stringsAsFactors = TRUE) #Y
summary(properties_2016)

###Converting Variables
properties_2016$basementsqft[is.na(properties_2016$basementsqft)] <- 0
properties_2016$basementsqft <- ifelse(properties_2016$basementsqft>0,1,0)
properties_2016$basementsqft <- as.factor(properties_2016$basementsqft)
properties_2016$garagetotalsqft[is.na(properties_2016$garagetotalsqft)] <- 0
properties_2016$garagetotalsqft<- ifelse(properties_2016$garagetotalsqft>0,1,0)
properties_2016$garagetotalsqft <- as.factor(properties_2016$garagetotalsqft)
properties_2016["age"] <- 2016-properties_2016["yearbuilt"]

###Dropping Variables with more than 60% of N/A values
null_percent <- map_dbl(properties_2016, function(x) {round((sum(is.na(x))/length(x))*100,2)})
lessthan60 <- null_percent[null_percent < 60]
properties <- properties_2016[,c(names(lessthan60))]

###Combining X features and Y targets into one file
dataset <- merge(properties, train_2016,by="parcelid",all.y=TRUE)
dim(dataset) #90275
dataset <- subset(dataset, select=-c(propertyzoningdesc, regionidcounty, regionidzip, regionidcity, censustractandblock, hashottuborspa, assessmentyear, fips, yearbuilt, latitude, longitude, transactiondate, propertycountylandusecode, parcelid))
dataset<-na.omit(dataset)
dim(dataset) #50635



###Multicollinearity
corrplot(cor(dataset[, unlist(lapply(dataset, is.numeric))]))
sum(dataset$structuretaxvaluedollarcnt + dataset$landtaxvaluedollarcnt == dataset$taxvaluedollarcnt)
sum(dataset$calculatedfinishedsquarefeet == dataset$finishedsquarefeet12)
sum(dataset$bathroomcnt == dataset$calculatedbathnbr)
dataset <- subset(dataset, select=-c(structuretaxvaluedollarcnt, bedroomcnt, landtaxvaluedollarcnt, taxamount, finishedsquarefeet12, calculatedbathnbr, fullbathcnt, rawcensustractandblock))
corrplot(cor(dataset[, unlist(lapply(dataset, is.numeric))]))

hist(dataset$logerror, breaks=70, xlim=c(-4,4))
mean(dataset$logerror)
sd(dataset$logerror)
ggplot(dataset,aes(logerror~calculatedfinishedsquarefeet)) + geom_point()

###Lasso####
set.seed(123)
x <- sparse.model.matrix(logerror~.,dataset)[,-1]
y <- dataset$logerror
train <- sample(1:nrow(x), nrow(x)*.7)
test <- (-train)
#creating train and test indices
ytest <- y[test]
grid <- 10^seq(from=10, to=-2, length=100)
#creates lambda grid
cv.outlasso <- cv.glmnet(x[train, ], y[train], alpha=1, lambda=grid)
#perform CV to get best model
lassolam <- cv.outlasso$lambda.min
#obtain best lambda from CV
lassomodel <- glmnet(x[train, ], y[train], alpha=1, lambda=lassolam)
#create lasso model with best lambda
lassopred <- predict(lassomodel, s=lassolam, newx=x[test,])
#make predictions with lasso model
(lassoMSE <- mean((lassopred-y[test])^2))
#find the test MSE of the lasso model
coef(lassomodel)



###Regular Boosting Model####
set.seed(123)
trainIndex <- sample(1:nrow(dataset), size=nrow(dataset)*.7)
#creates train rows to subset data into train and test
boostmod <- gbm(logerror~., data=dataset[trainIndex,], n.trees = 500, shrinkage=.5, distribution="gaussian")
preds <- predict.gbm(boostmod, dataset[-trainIndex,], n.trees=100)
(boostmse <- mean((preds-dataset$logerror[-trainIndex])^2))
summary(boostmod, plotit=TRUE)[1:5,1:2] #Top 5 Variables



###Random Forest
#rfmod <- randomForest(logerror~., data=dataset, subset=trainIndex, mtry=6, importance=TRUE)
#yhat<-predict(rfmod, newdata=dataset[-trainIndex,])
#(rfMSE <- mean((dataset$logerror[-trainIndex]-yhat)^2))
rfMSE = 0.02859478
varImpPlot(rfmod, n.var=5)




###xgBoost Model - Tuned#### **
set.seed(123)
train <- dataset[trainIndex, ]
test <- dataset[-trainIndex, ]
#splits data into train and test
train_x <- sparse.model.matrix(logerror~.,train)[,-1]
#turns train into a data matrix
train_y <- train$logerror
test_x <- sparse.model.matrix(logerror~.,test)[,-1]
#turns test into a data matrix
test_y <- test$logerror
trainx <- xgb.DMatrix(train_x, label=train_y)
testx <- xgb.DMatrix(test_x, label=test_y)
eta = c(.05, .07, .1, .3)
gamma = c(1, 5, 7, 10)
tunemses <- c()
for (i in eta){
  for (j in gamma){
  #turning matrix into matrices that can be used by xGboost
    xgmod <- xgboost(trainx, eta=i, objective="reg:squarederror", nrounds=100, gamma=j, verbose=F)
    #creating an xGboost model
    xgpred <- predict(xgmod, testx)
    #making predictions with the xGmodel
    (xGmse <- mean((test_y-xgpred)^2))
    #getting the test MSE of the xG model
    tunemses <- rbind(tunemses, c(xGmse, i, j))
  }
}
besteta = tunemses[which.min(tunemses),2]
bestgamma = tunemses[which.min(tunemses),3]
xgmod <- xgboost(trainx, eta=besteta, objective="reg:squarederror", nrounds=100, gamma=bestgamma, verbose=F)
xgpred <- predict(xgmod, testx)
#making predictions with the xGmodel
(xGmse <- mean((test_y-xgpred)^2))
#getting the test MSE of the xG model
xgb.ggplot.importance(xgb.importance(colnames(dataset[,-length(colnames(dataset))]), model = xgmod), top_n = 5)

#TESTING####
###Backwards Selection
library(leaps)
regfit.fwd<-regsubsets(logerror~., dataset, nvmax=dim(dataset)[2], method="forward")
regfit.summary<-summary(regfit.fwd)
print(regfit.summary$adjr2)
which.max(regfit.summary$adjr2) 
options(scipen = 999)
print(coef(regfit.fwd, 12))

###xgBoost Model#### **
set.seed(123)
train <- dataset[trainIndex, ]
test <- dataset[-trainIndex, ]
#splits data into train and test
train_x <- sparse.model.matrix(logerror~.,train)[,-1]
#turns train into a data matrix
train_y <- train$logerror
test_x <- sparse.model.matrix(logerror~.,test)[,-1]
#turns test into a data matrix
test_y <- test$logerror
trainx <- xgb.DMatrix(train_x, label=train_y)
testx <- xgb.DMatrix(test_x, label=test_y)
#turning matrix into matrices that can be used by xGboost
xgmod <- xgboost(trainx, eta=.1, objective="reg:squarederror", nrounds=100, max.depth=2, verbose=F)
#creating an xGboost model
xgpred <- predict(xgmod, testx)
#making predictions with the xGmodel
(xGmse <- mean((test_y-xgpred)^2))
#getting the test MSE of the xG model
xgb.plot.importance(xgb.importance(colnames(dataset[,-length(colnames(dataset))]), model = xgmod))
#plot most important features
