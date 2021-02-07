######################### Assignment Handwritten Digit Recognition #########################
######################### Submitted by Mukul Tamta #########################################

# Loading libraries

library(ggplot2)
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(gridExtra)

#Importing MNIST datasets

train_mnist <- read.csv("mnist_train.csv",header = FALSE)
test_mnist <- read.csv("mnist_test.csv",header = FALSE)


#Structure of the train_mnist dataset

str(train_mnist)
str(test_mnist)


# Naming the headers in both the imported datasets

colnames(train_mnist)[1] <- c("digit")
colnames(train_mnist)[2:785] <- c(1:784)

colnames(test_mnist)[1] <- c("digit")
colnames(test_mnist)[2:785] <- c(1:784)


#printing first few rows of train_mnist

head(train_mnist)

#Exploring the train_mnist dataset

summary(train_mnist)

#Check for NA values in train_mnist and test_mnist dataset

sapply(train_mnist,function(x) sum(is.na(x)))
sum(is.na(train_mnist))

sapply(test_mnist,function(x) sum(is.na(x)))
sum(is.na(test_mnist))

#No NA values were found in both train_mnist and test_mnist

#Changing digit column to factor in train_mnist and test_mnist

train_mnist$digit<-factor(train_mnist$digit)
test_mnist$digit<-factor(test_mnist$digit)

# Take only 15% of the Train data from Train MNIST dataset for model preparation (As advised by Upgrad)

set.seed(100)
train1.indices = sample(1:nrow(train_mnist), 0.15*nrow(train_mnist))
train1 = train_mnist[train1.indices, ]

# As per comments of TA in Discussion Forum (https://learn.upgrad.com/v/course/163/question/100305)
# For training and cross-validation, I have divided the sample taken from the training dataset(train1-9000 observations) in 70-30 ratio.

set.seed(100)
train2.indices = sample(1:nrow(train1), 0.7*nrow(train1))
train2 = train1[train2.indices, ]
cvdn = train1[-train2.indices, ]


# Constructing Model with train2 (6300 observations) and testing it on cvdn (2700 observations)

#Using Linear Kernel Model
Model_linear <- ksvm(digit~ ., data = train2, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, cvdn)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,cvdn$digit)

# In Linear Model 0.9107 Accuracy is found
# The Min. Sensitivity found among all classes is 0.85556
# The Min. Specificity found among all classes is 0.98197

#Using RBF Kernel Model
Model_RBF <- ksvm(digit~ ., data = train2, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, cvdn)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,cvdn$digit)

# In RBF Kernel model 0.9589 Accuracy is found
# The Min Sensitivity found among all classes is 0.92193
# The Min Specificity found among all classes is 0.99259

# Clearly RBF model seems to be a better model over here then the Linear model .So we would use RBF model.


# Finding the value of C and sigma from RBF model

Model_RBF

#Hyperparameter : sigma =  1.62642402888486e-07
#parameter : cost C = 1


############   Hyperparameter Tuning and Cross Validation #####################

# method =  CV means  Cross Validation.
# Number = 5 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"


# I have taken different values of sigma and Cost C on the basis of the values of sigma =  1.62642402888486e-07 and C= 1

set.seed(100)
grid <- expand.grid(.sigma=c(0.62642402888486e-07,1.62642402888486e-07,2.62642402888486e-07,3.62642402888486e-07), .C=c(0.5,1,2,3))


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
#trcontrol = Our traincontrol method.

fit.svm <- train(digit~., data=train2, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)


#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 3.626424e-07 and C = 3.

#Accuracy was found to be equal to 0.9626991 when sigma = 3.626424e-07 and C = 3

# Rebuild the RBF model with values sigma = 3.626424e-07 and C = 3 and test it with 2700 observations(cvdn)

Tuned_Model_RBF <- ksvm(digit~ ., data = train2, scale = FALSE, kernel = "rbfdot",C=3,kpar= list(sigma=3.626424e-07))
Tuned_Model_RBF

New_Eval_RBF<- predict(Tuned_Model_RBF, cvdn)

confusionMatrix(New_Eval_RBF,cvdn$digit)

# Accuracy has increased from 0.9589 to 0.9737
# The Min. Sensitivity found among all classes is 0.94796 . This is better then the previous model.
# The Min. Specificity found among all classes is 0.99465 . This is better then the previous model.
# Hence clearly the Tuned model has increased Accuracy,Sensitivity & Specifity.

#######################################################################################################
#######Using the Tuned model built above on Test Dataset(test_mnist) containing 10000 observations #######


Eval_RBF_Test<- predict(Tuned_Model_RBF, test_mnist)

confusionMatrix(Eval_RBF_Test,test_mnist$digit)

# The Tuned model's Accuracy on test_mnist dataset is found 0.9647
# The Min. Sensitivity found among all classes is 0.9358
# The Min. Specificity found among all classes is 0.9943

#Overall Statistics

#Accuracy : 0.9647          
#95% CI : (0.9609, 0.9682)
#No Information Rate : 0.1135          
#P-Value [Acc > NIR] : < 2.2e-16       
#Kappa : 0.9608          
#Mcnemar's Test P-Value : NA

######################END OF ASSIGNMENT##################################