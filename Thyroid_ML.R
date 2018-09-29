
set.seed(11)
tmp = "http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/"

##load and inspect training dataset
train = "ann-train.data"
train = read.table(paste(tmp, train, sep = ""))
head(train)

##plot few different variables against the class for training dataset
library(ggplot2)
ggplot(train, aes(x = V1, y = V17)) + geom_point(aes(colour=V22), size = 3) + xlab("v1") + ylab("v17") + ggtitle("Class vs v1 and v17")
ggplot(train, aes(x = V18, y = V19)) + geom_point(aes(colour=V22), size = 3) + xlab("v18") + ylab("v19") + ggtitle("Class vs v18 and v19")
ggplot(train, aes(x = V20, y = V21)) + geom_point(aes(colour=V22), size = 3) + xlab("v20") + ylab("v21") + ggtitle("Class vs v20 and v21")

##create input matrix for training dataset
x_train = train[,1:21]
normalizeData(x_train, type = "norm")
head(x_train)

##create output vector for training dataset
library(RSNNS)
y_train=decodeClassLabels(train[,22])
head(y_train)

##load and inspect testing dataset
tst = "ann-test.data"
tst = read.table(paste(tmp, tst, sep = ""))
head(tst)

##create input matrix for testing dataset
x_test=tst[,1:21]
normalizeData(x_test, type = "norm")
head(x_test)

##create output vector for testing dataset
y_test=decodeClassLabels(tst[,22])
head(y_test)

##run multi-layer perceptron model with backpropagation
model_1 = mlp(x_train, y_train, size = c(3), maxit=500, learnFunc= "Std_Backpropagation", learnFuncParams=c(0.1), inputsTest=x_test, targetsTest=y_test)

##plot error against no of epochs
plotIterativeError(model_1)

## print details of the model
model_1

##generate confusion matrix for training dataset
confusionMatrix(y_train,fitted.values(model_1))

##generate confusion matrix for testing dataset
predictions <- predict(model_1,x_test)
confusionMatrix(y_test,predictions)

##run multi-layer perceptron model with batched backpropagation
model_2 = mlp(x_train, y_train, size = c(3), maxit=500, learnFunc= "BackpropBatch", learnFuncParams=c(0.01), inputsTest=x_test, targetsTest=y_test)

##plot error against no of epochs
plotIterativeError(model_2)

## print details of the model
model_2

##generate confusion matrix for training dataset
confusionMatrix(y_train,fitted.values(model_2))

##generate confusion matrix for testing dataset
predictions <- predict(model_2,x_test)
confusionMatrix(y_test,predictions)

##run multi-layer perceptron model with quickprop
model_3 = mlp(x_train, y_train, size = c(3), maxit=500, learnFunc= "Quickprop", learnFuncParams=c(0.01), inputsTest=x_test, targetsTest=y_test)

##plot error against no of epochs
plotIterativeError(model_3)

## print details of the model
model_3

##generate confusion matrix for training dataset
confusionMatrix(y_train,fitted.values(model_3))

##generate confusion matrix for testing dataset
predictions <- predict(model_3,x_test)
confusionMatrix(y_test,predictions)

##run multi-layer perceptron model with Rprop
model = mlp(x_train, y_train, size = c(3), maxit=500, learnFunc= "Rprop", learnFuncParams=c(0.01), inputsTest=x_test, targetsTest=y_test)

##plot error against no of epochs
plotIterativeError(model)

## print details of the model
model

##generate confusion matrix for training dataset
confusionMatrix(y_train,fitted.values(model))

##generate confusion matrix for testing dataset
predictions <- predict(model,x_test)
confusionMatrix(y_test,predictions)
