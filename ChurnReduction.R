#remove all the objects stored
rm(list=ls())
#set current working directory
setwd("C:/Users/SHRAVYA/Desktop/edwisor/project 2")
#install packages
install.packages(c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information", "MASS", "rpart", "gbm", "ROSE", "sampling", "DataCombine", "inTrees"))
library(readxl)     # Super simple excel reader
library(mice)       # missing values imputation
library(naniar)     # visualize missing values
library(dplyr)
library(corrplot)
library(ggplot2)
library(tidyverse)
library(randomForest)
library(caret)
library(data.table)
library(Boruta)
library(rpart)
## Read the data
Train_data = read.csv("Train_data.csv", header = T, na.strings = c(" ", "", "NA"))
Test_data = read.csv("Test_data.csv", header = T, na.strings = c(" ", "", "NA"))
#############################exploratory data analysis##############################
#studying the structure of the given dataset
str(Train_data)
#As we can see, few variables have a wrongly placed datatype.
#The variable , phone.number is actually meant to be a continuoos variable but it is analysed as a factor variable with 3333 levels. hence it has to be altered
#The variable area.code has to be a factor variable.
Train_data$phone.number  = as.numeric(Train_data$phone.number )
Test_data$phone.number  = as.numeric(Test_data$phone.number )
Train_data$area.code  = as.factor(Train_data$area.code )
Test_data$area.code  = as.factor(Test_data$area.code )
str(Train_data)
#################################missing value analysis###########################
missing_val = data.frame(apply(Train_data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(Train_data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "Missing_perc.csv", row.names = F)
sum(is.na(Train_data))
sum(is.na(Test_data))
#as we can see, there are no missing values. Hence no missing value imputation is requirede.calls, Churn)
############################################variable analysis#########################
library(ggplot2)
library(ggpubr)
theme_set(theme_pubr())
ggplot(Train_data, aes(Churn)) +
  geom_bar(fill = "#0073C2FF") +
  theme_pubclean()
#outlier analysis
numeric_index = sapply(Train_data,is.numeric) #selecting only numeric

numeric_data = Train_data[,numeric_index]

cnames = colnames(numeric_data)
 
 for (i in 1:length(cnames))
 {
assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(Train_data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
            geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                         outlier.size=1, notch=FALSE) +
            theme(legend.position="bottom")+
            labs(y=cnames[i],x="responded")+
            ggtitle(paste("Box plot of churned for",cnames[i])))
}

# ##Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)
#feature selection
################################### Correlation Plot - to check multicolinearity between continous variables
library(corrgram)
corrgram(Train_data[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
#few independent variables are found to be highly correlated to each other. hence one of each copy is eliminated
## Chi-squared Test of Independence-to check the multicolinearity between categorical variables
factor_index = sapply(Train_data,is.factor)
factor_data = Train_data[,factor_index]
for (i in 1:5)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}
#the variable area code has a p value of greater than 0.05. Hence we eliminate the variable
## Dimension Reduction
Train_data = subset(Train_data, 
                         select = -c(area.code,total.day.minutes,total.eve.minutes, total.night.minutes, total.intl.minutes))
Test_data = subset(Test_data, 
                    select = -c(area.code,total.day.minutes,total.eve.minutes, total.night.minutes, total.intl.minutes))
###################################model developement###############################
#Clean the environment
library(DataCombine)
rm(list= ls()[!(ls() %in% c('Train_data','Test_data'))])
##Decision tree for classification
#Develop Model on training data
install.packages("C50")
library(C50)
C50_model = C5.0(Churn ~., Train_data, trials = 100, rules = TRUE)
#Summary of DT model
summary(C50_model)
#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")
#Lets predict for test cases
C50_Predictions = predict(C50_model, Test_data[,-16], type = "class")
##Evaluate the performance of classification model
library(caret)
ConfMatrix_C50 = table(Test_data$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)
#accuracy=95.86%
#false negetive rate=27.67%
########################################Random Forest####################################
library(randomForest)
RF_model = randomForest(Churn ~ ., Train_data, importance = TRUE, ntree = 1000)
#Extract rules fromn random forest
#transform rf object to an inTrees' format
library(RRF)
library(inTrees)
treeList = RF2List(RF_model)  
# 
# #Extract rules
exec = extractRules(treeList, Train_data[,-16])  # R-executable conditions
# #Make rules more readable:
readableRules = presentRules(exec, colnames(Train_data))
# #Get rule metrics
ruleMetric = getRuleMetric(exec, Train_data[,-16], Train_data$Churn)  # get rule metrics
#Presdict test data using random forest model
RF_Predictions = predict(RF_model, Test_data[,-16])

##Evaluate the performance of classification model
ConfMatrix_RF = table(Test_data$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy = 94.66
#FNR = 31.25
#Logistic Regression
logit_model = glm(Churn ~ ., data = Train_data, family = "binomial")

#summary of the model
summary(logit_model)

#####################################predict using logistic regression##################################
logit_Predictions = predict(logit_model, newdata = Test_data, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_LR = table(Test_data$Churn, logit_Predictions)
confusionMatrix(ConfMatrix_LR)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy: 90.89
#FNR: 67.85
set.seed(4543)
Train.rf <- randomForest(Churn ~ ., data=Train_data, ntree=1000, keep.forest=FALSE,
                          importance=TRUE)
varImpPlot(Train.rf)
