##################################################
# ECON 418-518 Homework 3
# Constant Haden
# The University of Arizona
# haden1@arizona.edu 
# 17 November 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set seed
set.seed(418518)

# load in data and assign it to a variable
data <- read.csv("ECON_418-518_HW3_Data.csv")

#####################
# Problem 1
#####################


#################
# Question (i)
#################

# remove columns “fnlwgt”, “occupation”, “relationship”, “capital.gain”,
# “capital.loss”, and “educational.num” from the data
data2 <- subset(data, select = -c(fnlwgt, occupation, relationship,
                                  capital.gain, capital.loss, educational.num))

#################
# Question (ii)
#################

# a: income becomes 1 if income > 50k, and 0 if otherwise
data2$income <- ifelse(data2$income == ">50K", 1, 0)

# b: race becomes 1 if equal to "White" and 0 if otherwise
data2$race <- ifelse(data2$race == "White", 1, 0)

# c: gender becomes 1 if equal to "Male" and 0 if otherwise
data2$gender <- ifelse(data2$gender == "Male", 1, 0)

# d: workclass becomes 1 if equal to "Private" and 0 if otherwise
data2$workclass <- ifelse(data2$workclass == "Private", 1, 0)

# e: native.country becomes 1 if equal to "United-States" and 0 if otherwise
data2$native.country <- ifelse(data2$native.country == "United-States", 1, 0)

# f: marital.status becomes 1 if equal to "Married-civ-spouse" and 0 if otherwise
data2$marital.status <- ifelse(data2$marital.status == "Married-civ-spouse", 1, 0)

# g: education becomes 1 if equal to "Bachelors", "Masters", or "Doctorate" and
# 0 if otherwise
data2$education <- ifelse(data2$education == "Bachelors" | data2$education ==
                            "Masters" | data2$education == "Doctorate", 1, 0)

# h: new variable "age_sq", which is age squared
age_sq <- data2$age^2
# add age_sq to data table
data2$age_sq <- age_sq

# load in data manipulation package
library(dplyr)

# i: standardize age, age_sq, and hours per week worked
data3 <- data2 %>% mutate_at(c('age', 'age_sq', 'hours.per.week'),
                             ~(scale(.) %>% as.vector))


#################
# Question (iii)
#################

# return a sum of how many "1"s appear in each column
colSums(data3 == 1)

# return how many values in the data table are "?", or N/A
colSums(data == "?")

# convert income to factor type
as.factor(data3$income)

################
# Question (iv)
################

# Calculate the number of rows for the training set (70% of the total rows)
train_size <- floor(nrow(data3) * 0.70)

# create the training set from the beginning to the .70 floor
train_set <- data3[1:train_size, ]
# create the test set from the .70 floor to the end
test_set <- data3[(train_size + 1):nrow(data3), ]

################
# Question (v)
################

# install the necessary packages
install.packages("caret")
# glmnet for the lasso regression
install.packages("glmnet")

# load in previously installed packages
library(caret)
library(glmnet)

train_set$income <- as.numeric(train_set$income)

# define trainControl with 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                              summaryFunction = twoClassSummary))

# create a sequence of lambda values from 10^5 to 10^(-2) over 50 values
lambda_grid <- 10^seq(5, -2, length = 50)

# define the lasso model
lasso_model <- train( income ~ ., data = train_set, method = "glmnet",
  trControl = train_control, tuneGrid = 
    expand.grid(alpha = 1,lambda = lambda_grid, metric = "Accuracy")
)
# print the results
print(lasso_model)
# create a table for the coefficients of each variable
coeffs <- coef(lasso_model$finalModel, s =lasso_model$bestTune$lambda)
# print that table for analysis
print(coeffs)

# e:
# new training set without race and gender, the variables closest to zero
train_set2 <- subset(train_set, select = -c(race, gender))

lasso_model_two <- train(
  income ~ ., data = train_set2, method = "glmnet", trControl = train_control,
  # alpha = 1 to indicate a lasso model
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid), metric = "Accuracy"
)
ridge_model <- train(
  income ~ ., data = train_set2, method = "glmnet", trControl = train_control,
  # alpha = 0 to indicate a ridge model
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid), metric = "Accuracy"
)


################
# Question (vi)
################

# install randomForest package
install.packages("randomForest")
# load package
library(randomForest)

# define the grid for different number of trees (100, 200,
# and 300) and random features at each split (2, 5, and 9)
ran_for_grid <- expand.grid(mtry = c(2, 5, 9), ntree = c(100, 200, 300))

# train model with 5-fold cross-validation
ran_for_model <- train(
  income ~ ., data = train_set, method = "rf",
  trControl = train_control, tuneGrid = ran_for_grid, metric = "Accuracy"
)
print(ran_for_model)

# i believe this is the actual correct notation. but, no matter what i do, i
# keep running into the same error: "vector memory limit of 16.0 Gb reached,
# see mem.maxVSize()"
rf <- randomForest(income~., data=train_set, proximity=TRUE)
print(rf)

# for the confusion matrix, the variable equal to data should be the predicted
# value, i.e, the results of the random forest model. the variable equal to
# reference should be the actual expected value, i.e, the training set
conMat <- confusionMatrix(data = rf, reference = train_set)


