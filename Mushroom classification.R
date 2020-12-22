install.packages("caret")
install.packages("pROC")
install.packages("mlbench")
install.packages("arules")
install.packages("e1071")
install.packages("rpart.plot")
install.packages('corrplot')

####################################################################################################
library(e1071)
library(caret)
library(pROC)
library(mlbench)
library(rpart)
library(rpart.plot)
library(corrplot)
####################################################################################################
## Import the dataset
# read data
mushroom <- read.csv(file.choose(), header = TRUE,stringsAsFactors = FALSE)

### EDA

# Check the structure of data
str(mushroom)

# summarize the data
print(summary(mushroom))

#find if mushroom dataset contain any missing values

sum(is.na(mushroom))

# load the columns of dataset

fields <- c("class",
            "cap_shape",
            "cap_surface",
            "cap_color",
            "bruises",
            "odor",
            "gill_attachment",
            "gill_spacing",
            "gill_size",
            "gill_color",
            "stalk_shape",
            "stalk_root",
            "stalk_surface_above_ring",
            "stalk_surface_below_ring",
            "stalk_color_above_ring",
            "stalk_color_below_ring",
            "veil_type",
            "veil_color",
            "ring_number",
            "ring_type",
            "spore_print_color",
            "population",
            "habitat")

colnames(mushroom) <- fields

# classify the datset according to classes

mushroom$class[mushroom$class == 'e'] <- 'Edible'
mushroom$class[mushroom$class == 'p'] <- 'Poisonous'

mushroom$class <- factor(mushroom$class)

#create a copy of mushroom data into DF 

DF <- as.data.frame(unclass(mushroom))

DF$veil_type = NULL

# get all features which are factors into convert variable
convert<-sapply(DF,is.factor)

# convert all the factor features into list using unclass function
d1<-sapply(DF[,convert],unclass)

#merge all categorical and non categorical features into mushrooms variable
mushrooms<-cbind(d1[,!convert],d1)
#convert mushrooms dataset matrix  format to data frame
x=as.data.frame(mushrooms)
# correlation of the mushroms dataset using cor() function
a <- cor(mushrooms)
#plot correlation plot using corrplot
corrplot(a, method = "circle", type = "upper")


#correlation test using chisquare t-test between gill attachment and veil color

chisq.test(mushroom$gill_attachment,mushroom$veil_color)

####################################################################################################

## Split the data set into 80% training and 20% Training
set.seed(1234)
# split the datset into 80% 20% samples
training_split = createDataPartition(y = mushroom$class, p = 0.80, list = FALSE);
# split trianing set
training_set = mushroom[training_split,]
training_set
# Split testing set
testing_set = mushroom[-training_split,]
testing_set1 = testing_set[-c(1)]
testing_set1
write.csv(testing_set, file = "test.csv")
####################################################################################################

## Decision tree model

# Generate the model
model = rpart(formula = class~.,data=training_set , method = 'class') 
# Print the model
print(model)

# Plot the decision tree
rpart.plot(model, extra = 1)
rpart.plot(model, extra = 4)

# Determine the Variable importance factor
x = varImp(model)
x

# Display the varable importance plot
ggplot(data = x, aes(x = row.names(x),y = x$Overall))+geom_bar(stat = "identity", col = "Green")+labs(title = "Bar plot", y = "count", x= "variables")+ coord_flip()

# Stacked plot for Odor 
ggplot(mushroom, aes(mushroom$odor, fill = mushroom$class))+geom_bar()+labs(title ="Stacked Bar Chart", x = "odor" , y = "count of odors", fill = "Class")

#Stacked plot for Spore print color
ggplot(mushroom, aes(mushroom$spore_print_color, fill = mushroom$class))+geom_bar()+labs(title ="Stacked Bar Chart", x = "spore print color" , y = "count of spore print color", fill = "Class")

#Testing our model built on training data on test data
trainPredDT = predict(model, newdata = training_set, type = "class")

# construct a confusion matrix
trainTableDT = table(training_set$class, trainPredDT)
trainTableDT

# predict for testing set
testPredDT=predict(model, newdata=testing_set1, type="class")
testPredDT
write.csv(testPredDT, file = "Decision_tree_Output.csv")

# construct a confusion matrix
testTableDT=table(testing_set$class, testPredDT)
print(testTableDT)

# Print accuracy
print("DT acuracy ")
print(mean(testPredDT == testing_set$class))

## End of decision tree
####################################################################################################

##  KNN 

KNN_TrainD = training_set # Use variable KNN_TrainD to load the data
KNN_TrainD$veil_type = NULL # Set the viel type null as it wont generate any classifications

# Build model for KNN
model_fit = train(class ~ ., method = "knn", data = KNN_TrainD, trControl = trainControl(method = 'cv', number = 3, classProbs = TRUE))

# Print model
print(model_fit)

# plot the model
plot(model_fit)

#Testing our model built on training data on test data
testPredKNN = predict(model_fit, newdata = testing_set1, type = "raw")

testTableKNN=table(testing_set$class, testPredKNN)
testTableKNN

# Print accuracy
print(" Knn accuracy ")
print(mean(testPredKNN == testing_set$class))

## End of KNN
####################################################################################################
## Naive Bayes

# Generate the model for Naive Bayes
model2 =naiveBayes(training_set[,c(2:22)],training_set$class)

# Print the model
print(model2)

# predicted labels for testing set
testPredNB=predict(model2, newdata=testing_set1, type="class")

# construct a confusion matrix
testTableNB=table(testing_set$class, testPredNB)
testTableNB

# Print accuracy
print("Naive bayes accuracy")
print(mean(testPredNB == testing_set$class))

## End of Naive Bayes

####################################################################################################

# plot a confusion matrix for Naive Bayes

fourfoldplot(testTableNB, color = c("#CC6666", "#99CC99"),conf.level = 0, margin = 1, main = "Confusion Matrix for Naive Bayes")

# plot a confusion matrix for Decision Tree

fourfoldplot(testTableDT, color = c("#CC6666", "#99CC99"),conf.level = 0, margin = 1, main = "Confusion Matrix for Decision Tree")

# plot a confusion matrix for KNN

fourfoldplot(testTableKNN, color = c("#CC6666", "#99CC99"),conf.level = 0, margin = 1, main = "Confusion Matrix for KNN")

######################################### End of Project ###########################################

