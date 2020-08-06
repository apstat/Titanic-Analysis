##### Objective #####
# Q1. How does features depend upon chance of Survival?
# Q2. Predicting the Survival using preferred features in Q1.
# Q3. Predicting the Survival in the entire ship.

library(titanic)
library(dplyr)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(tibble)

# Load train and test data
dat1 <- titanic_train
dat2 <- titanic_test
head(dat1)

data <- bind_rows(dat1,dat2)
# Finding missing values
colSums(is.na(data))
colSums(data == "")

# Cleaning & Processing the data
data_clean <- data %>% 
  mutate(Survived = factor(Survived),
         Pclass = factor(Pclass),
         Age = ifelse(is.na(Age),median(Age,na.rm = TRUE),Age),
         Embarked = ifelse(Embarked == "",'S',Embarked)%>%factor(),
         FamilySize = SibSp + Parch + 1) %>%
  select(Survived,Sex,Pclass,Age,SibSp,Parch,FamilySize,Fare,Embarked)

########------- Analysis of Q1 -------#########
# The train data has the Survived data, so here we will use only train data set
# The features are Sex, Pclass, FamilySize, Embarked
train_set <- data_clean[1:dim(titanic_train)[1],]
# (1) How well sex depends on chance of Survival?,i.e., Sex as a function of Survival
ggplot(train_set,aes(x = Sex,fill = Survived)) + 
  geom_bar(position = "fill") + ylab("Frequency")
# What proportion of train data Female & Male survived?
train_set %>% group_by(Sex) %>% summarize(Survived = mean(Survived == 1)*100)
# So, if the Sex = Female, then chance of Survival was greater about 74%.

# (2) How well Pclass depends on chance of Survival?,i.e., Pclass as a function of Survival
ggplot(train_set,aes(x = Pclass,fill = Survived)) + 
  geom_bar(position = "fill") + ylab("Frequency")
train_set %>% group_by(Pclass) %>% summarize(Survived = mean(Survived == 1)*100)
# So, if Passenger is from 1st class, chance of survival is greater, i.e., 63%.

# (3) How well Embarked depends on chance of Survival?, i.e., Embarked as a function of Survival
ggplot(train_set,aes(x = Embarked,fill = Survived)) + 
  geom_bar(position = "fill") + ylab("Frequency")
train_set %>% group_by(Embarked) %>% summarize(Survived = mean(Survived == 1)*100)
# So, Passenger embarked from 'C' has greater chance of Survival about 55%.

# (4) How well FamilySize depends on chance of Survival?, i.e., FamilySize as a function of Survival
ggplot(train_set,aes(x = FamilySize,fill = Survived)) +
  geom_bar(position = "fill") + ylab("Frequency")
train_set %>% group_by(FamilySize) %>% summarize(Survived = mean(Survived == 1)*100)
# So, FamilySize between 2 & 4 has more than 50% chance of Survival.

# (5) How well Age depends on chance of Survival?, i.e., Age as a funtion of Survival
ggplot(train_set,aes(x = Age,fill = Survived)) +
  geom_histogram(binwidth = 3,position = "fill") + ylab("Frequency")
train_set %>% group_by(Age) %>% summarize(Survived = mean(Survived == 1)*100)
# So, children less than 15y/o & old people >=80 has higher chance of survival.

##########-------- Analysis of Q2 --------##########
# Partition data into train & test sets
set.seed(42,sample.kind = 'Rounding')
index <- createDataPartition(train_set$Survived,times = 1,p = 0.5,list = FALSE)
train <- train_set[-index,]
test <- train_set[index,]

#######_:_:_:_:_: Logistic Regression :_:_:_:_:_#######
# (1) Since 'Sex' & 'Pclass' has greater than 60% chance of survival so predict using these two classes
model1 <- train(Survived~Sex+Pclass,method = 'glm',data = train)
pred1 <- predict(model1,test)
mean(pred1 == test$Survived)
# So, the mean of correct predictions is quite high about 78%.
# (2) Using all the above features together to predict on the test set
model2 <- train(Survived~.,method = 'glm',data = train)
pred2 <- predict(model2,test)
m2 <- mean(pred2 == test$Survived)
m2
# So, the mean of correct predictions has increased a little to about 80%.
# Creating Confusion Matrix for model2
cm2 <- confusionMatrix(factor(pred2),test$Survived)$byClass['F1']
cm2

#######_:_:_:_:_: Decision Tree :_:_:_:_:_#######
# Training a Decision Tree Model using `rpart`
model3 <- rpart(Survived~.,data = train,method = 'class')
pred3 <- predict(model3,test,type = 'class')
m3 <- mean(pred3 == test$Survived)
m3
# The mean of correct predictions is about 79%.
rpart.plot(model3)
# Creating Confusion Matrix for model3
cm3 <- confusionMatrix(factor(pred3),test$Survived)
cm3

#######_:_:_:_:_: Random Forest Model :_:_:_:_#######
model4 <- randomForest(Survived~.,data = train)
pred4 <- predict(model4,test)
m4 <- mean(pred4 == test$Survived)
m4
# The mean of correct predictions is about 81%.
varImp(model4) # Importance of Variables
# Creating Confusion Matrix for model4
cm4 <- confusionMatrix(factor(pred4),test$Survived)
cm4

val <- c(m2,m3,m4,cm2,cm3,cm4)
matrix(val,ncol = 2,nrow = 3,byrow = FALSE,
       dimnames = list(c("Logistic Regression","Decision Tree","Random Forest"),
                       c("Mean Predictions","F1 Score")))
## The mean of correct predictions is obtained highest for Random Forest Model about 81%.
# Lets predict `titanic_test` using model4
test_set <- data_clean[length(train_set)+1:1309,]
Survived_pred <- predict(model4,test_set)[1:418]
# Now, we will form a data frame wrt `titanic_test`
result <- add_column(titanic_test,
                     Survived = as.numeric(as.character(Survived_pred)),
                     .before = "Pclass")
head(result)
# Proportion of Survival of all Passengers
full_data <- bind_rows(titanic_train,result)
nrow(full_data)
mean(full_data$Survived == 1)*100
