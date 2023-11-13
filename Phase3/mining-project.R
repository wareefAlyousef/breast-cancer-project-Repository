library("tidyverse")
library("dplyr")
library("readr")
library("Hmisc")
library("outliers")
library("mlbench")
library("ggplot2")
library("lattice")
library("caret")
library("caTools")
library("corrplot")
library("reshape2")


data <- read_csv("C:/Users/warif/OneDrive/Desktop/mining project data/data.csv")

View(data)

head(data)

nrow(data)

ncol(data)

dim(data)

names(data)

str(data)

summary(data)

describe(data) 

data1 <- data
View(data1)

##  Data reduction
data1 <- subset(data1, select = -...33)
data1 <- subset(data1, select = -id)
data1 <- data1[, c(2:ncol(data1), 1)]
View(data1)


## frequency table
diagnosis.table <- table(data$diagnosis)
colors <- terrain.colors(2) 
# a pie chart for the diagnosis 
diagnosis.prop.table <- prop.table(diagnosis.table)*100
diagnosis.prop.df <- as.data.frame(diagnosis.prop.table)
pielabels <- sprintf("%s - %3.1f%s", diagnosis.prop.df[,1], diagnosis.prop.table, "%")
colors = c ("pink", "skyblue1")
pie(diagnosis.prop.table,
    labels=pielabels,  
    clockwise=TRUE,
    col=colors,
    border="gainsboro",
    radius=0.8,
    cex=0.8, 
    main="Frequency Of Cancer Diagnosis")
legend(1, .4, legend=diagnosis.prop.df[,1], cex = 0.7,fill = colors)

#Remove the first  column
data2 <- data[, -1]
#Remove the last column
data2 <- data2[,-32]
#Tidy the data
#bc_data$diagnosis <- as.factor(bc_data$diagnosis)
summary(data2)
head(data2)
View(data2)


#corlrrelation
c <- cor(data2[,2:31])
corrplot(c, order = "hclust", tl.cex = 0.7)

#covariance 
cov(data2[, 2:31])

#Comparing the raidus, area and concavity of benging malignant stage
ggplot(data2, aes(x=diagnosis, y=radius_mean))+geom_boxplot()+ggtitle("area of bengin Vs malignant")
ggplot(data2, aes(x=diagnosis, y=concavity_mean))+geom_boxplot()+ggtitle("concavity of bengin Vs malignant")
ggplot(data2, aes(x=diagnosis, y=radius_worst))+geom_boxplot()+ggtitle("area of bengin Vs malignant")

# Reshape the data
data_long <- data2 %>%
  pivot_longer(cols = -diagnosis, names_to = "variable", values_to = "value")

# Plot box plots with separate y-axis scales and arranged in a grid
ggplot(data_long, aes(x = diagnosis, y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, nrow = 5, ncol = 6, scales = "free_y") +
  ggtitle("Box Plot for All Columns with Diagnosis") 


#Break up columns into groups, according to their suffix designation 
#(_mean, _se,and __worst) to perform visualisation plots off.
data_mean <- data[ ,c("diagnosis", "radius_mean", "texture_mean","perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean" )]

data_se <- data[ ,c("diagnosis", "radius_se", "texture_se","perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se" )]

data_worst <- data[ ,c("diagnosis", "radius_worst", "texture_worst","perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst" )]

#Plot histograms of "_mean" variables group by diagnosis
ggplot(data = melt(data_mean, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales =      'free_x')



#Plot histograms of "_se" variables group by diagnosis
ggplot(data = melt(data_se, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')


#Plot histograms of "_worst" variables group by diagnosis
ggplot(data = melt(data_worst, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')


#### Data preprocessing

## Data cleaning
#Checking NULL, FALSE means no null, TRUE cells means the value of the cell is null
is.na(data1)
# to find the total null values in the dataset
sum(is.na(data1))

# outliers
data2 <- subset(data1, select = -diagnosis)
# Function to identify outliers using z-score
find_outliers <- function(column) {
  median_val <- median(column)
  mad_val <- mad(column)
  modified_z_scores <- 0.6745 * (column - median_val) / mad_val
  return(which(abs(modified_z_scores) > 3))
}
# Function to count outliers for each attribute
count_outliers <- function(data) {
  outlier_counts <- sapply(data, function(col) length(find_outliers(col)))
  return(outlier_counts)
}
# Apply the count_outliers function to the data frame
outlier_counts <- count_outliers(data2)
# Print the outlier counts for each attribute
cat("Outlier counts for each attribute:\n")
print(outlier_counts)
# Create an empty list to store outlier indices for each attribute
outlier_indices <- list()
# Apply the find_outliers function to each attribute in the data frame
for (col in names(data2)) {
  outlier_indices[[col]] <- find_outliers(data2[[col]])
}
# Print the outlier indices
print(outlier_indices)
# Flatten the outlier indices into a single vector
flattened_indices <- unlist(outlier_indices)
# Get the outlier counts
outlier_counts <- table(flattened_indices)
# Print the outlier counts
cat("Outlier counts:\n")
print(outlier_counts)
# Get the total count of outliers
total_outliers <- sum(outlier_counts)
# Print the total count of outliers
cat("Total count of outliers:", total_outliers, "\n")
# Get the most frequent outlier objects 
most_frequent_outlier_objects <- if (length(flattened_indices) > 0) {
  table(flattened_indices)
} else {
  NULL
}
# Sort the most frequent outlier objects in descending order
sorted_most_frequent_outliers <- sort(most_frequent_outlier_objects, decreasing = TRUE)
# Get the outlier objects with a frequency greater than 5
frequent_outliers <- sorted_most_frequent_outliers[sorted_most_frequent_outliers > 5]
# Print the outlier objects with their frequencies
cat("Outlier objects with a frequency greater than 5:\n")
for (row in names(frequent_outliers)) {
  cat("Object", row, "- Frequency:", frequent_outliers[row], "\n")
}
# Identify the objects to be deleted
objects_to_delete <- as.integer(names(frequent_outliers))
# Delete the objects from the data frame
data1 <- data1[-objects_to_delete, ]
# Print the updated data frame
print(data1)


## Attribute transformation
data1$diagnosis <- ifelse(data1$diagnosis == "M", 1, ifelse(data1$diagnosis == "B", 0, data1$diagnosis))
# Convert the target variable to numeric or logical
data1$diagnosis <- as.numeric(data1$diagnosis)
View(data1)


## Feature scaling
#scale, with default settings, will calculate the mean and standard deviation of the entire vector, then "scale" each element by those values by subtracting the mean and dividing by the sd
data1 [, 1:30] = scale(data1 [, 1:30])
View(data1)


## Normalization
# Define function normalize()
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
cols_to_normalize <- 1:(ncol(data1) - 1)  # Select all columns except the last one
data1[, cols_to_normalize] <- lapply(data1[, cols_to_normalize], normalize)
View(data1)



## Discretization
# Number of bins for discretization
num_bins <- 20
# Function to perform discretization using equal width
equal_width_discretize <- function(var, num_bins) {
  breaks <- seq(min(var, na.rm = TRUE), max(var, na.rm = TRUE), length.out = num_bins + 1)
  cut(var, breaks = breaks, labels = FALSE)
}
# Subset the dataset with selected columns
cols_to_Discretiz <- 1:(ncol(data1) - 1)  # Select all columns except the last one
# Apply equal width discretization to selected columns
data1_discretized <- lapply(data1[, cols_to_Discretiz], equal_width_discretize, num_bins)
# Replace null values with 0
data1_discretized <- lapply(data1_discretized, function(x) ifelse(is.na(x), 0, x))
# Assign discretized values back to the original dataset
data1[, cols_to_Discretiz] <- data1_discretized
# View the modified dataset
View(data1)


## Feature selection
# ensure results are repeatable
set.seed(7)
# prepare training scheme
control4 <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
# train the model
model4 <- train(diagnosis ~.,data = data1 , method="lm", preProcess="scale", trControl=control4)
# estimate variable importance
importance4 <- varImp(model4 , scale=FALSE)
# plot importance
plot(importance4)
# train the model
model4 <- lm(diagnosis ~.,data = data1)
# estimate variable importance
importance4 <- varImp(model4 , scale=FALSE)
# Copy the importance3 dataframe to a new dataframe
importance4_copy <- importance4
# Add a column for row names
importance4_copy$name <- row.names(importance4_copy)
# Sort the dataframe based on the "Overall" column in descending order
importance4_copy <- importance4_copy[order(importance4_copy$Overall, decreasing = TRUE), ]
# Reset row names
row.names(importance4_copy) <- NULL
# Print the rearranged dataframe
print(importance4_copy) 
# Remove the least important 9 attributes
data1 <- subset(data1, select = -symmetry_se)
data1 <- subset(data1, select = -fractal_dimension_mean)
data1 <- subset(data1, select = -texture_mean)
data1 <- subset(data1, select = -`concave points_se`)
data1 <- subset(data1, select = -perimeter_se)
data1 <- subset(data1, select = -symmetry_mean)
data1 <- subset(data1, select = -compactness_se)
data1 <- subset(data1, select = -smoothness_mean)
data1 <- subset(data1, select = -compactness_worst)
View(data1)

#Data before preprocessing
rawData <- data
View(rawData)

#Preprocessed data
preprocessedData <- data1
View(preprocessedData)


colnames(preprocessedData)[6] <- "concave_points_mean"
colnames(preprocessedData)[19] <- "concave_points_worst"


# Classification
# Convert the response variable to a factor
preprocessedData$diagnosis <- factor(preprocessedData$diagnosis, levels = c(0, 1))
# Load required libraries
library(party)
library(FSelector)
library(partykit)
library(caret)

# Set seed for reproducibility
set.seed(7)

# Different partition methods (70-30, 60-40, and 80-20 splits)
ind1 <- sample(2, nrow(preprocessedData), replace=TRUE, prob=c(0.7, 0.3))
ind2 <- sample(2, nrow(preprocessedData), replace=TRUE, prob=c(0.6, 0.4))
ind3 <- sample(2, nrow(preprocessedData), replace=TRUE, prob=c(0.8, 0.2))

trainData1 <- preprocessedData[ind1 == 1,]
testData1 <- preprocessedData[ind1 == 2,]

trainData2 <- preprocessedData[ind2 == 1,]
testData2 <- preprocessedData[ind2 == 2,]

trainData3 <- preprocessedData[ind3 == 1,]
testData3 <- preprocessedData[ind3 == 2,]

# Convert diagnosis to a factor variable
trainData1$diagnosis <- factor(trainData1$diagnosis, levels = c(0, 1))
trainData2$diagnosis <- factor(trainData2$diagnosis, levels = c(0, 1))
trainData3$diagnosis <- factor(trainData3$diagnosis, levels = c(0, 1))

# Ensure consistent factor levels in training and test data
testData1$diagnosis <- factor(testData1$diagnosis, levels = c(0, 1))
testData2$diagnosis <- factor(testData2$diagnosis, levels = c(0, 1))
testData3$diagnosis <- factor(testData3$diagnosis, levels = c(0, 1))

#Information Gain method for selecting attributes using C50 algorithm
# Load required library for Information Gain
library(C50)

# Set seed for reproducibility
set.seed(7)

# Define the formula for the decision tree
formula_info_gain <- diagnosis ~ .

# Fit decision tree models using Information Gain
# Build decision tree model using Information Gain selected attributes (70-30 split)
model_info_gain1 <- C5.0(formula_info_gain, data = trainData1)
# Build decision tree model using Information Gain selected attributes (60-40 split)
model_info_gain2 <- C5.0(formula_info_gain, data = trainData2)
# Build decision tree model using Information Gain selected attributes (80-20 split)
model_info_gain3 <- C5.0(formula_info_gain, data = trainData3)

# Plot the decision trees for different partition methods (Information Gain)
# Plot decision tree for (70-30 split)
plot(model_info_gain1, main = "Information Gain Decision Tree (70-30 split)")
# Plot decision tree for (60-40 split)
plot(model_info_gain2, main = "Information Gain Decision Tree (60-40 split)")
# Plot decision tree for (80-20 split)
plot(model_info_gain3, main = "Information Gain Decision Tree (80-20 split)")

# Make predictions
#(70-30 split)
predictions_info_gain1 <- predict(model_info_gain1, testData1)
#(60-40 split)
predictions_info_gain2 <- predict(model_info_gain2, testData2)
#(80-20 split)
predictions_info_gain3 <- predict(model_info_gain3, testData3)

# Evaluate the models accuracy
#(70-30 split)
accuracy_info_gain1 <- sum(predictions_info_gain1 == testData1$diagnosis) / nrow(testData1)
#(60-40 split)
accuracy_info_gain2 <- sum(predictions_info_gain2 == testData2$diagnosis) / nrow(testData2)
#(80-20 split)
accuracy_info_gain3 <- sum(predictions_info_gain3 == testData3$diagnosis) / nrow(testData3)

# Function to calculate Precision, Sensitivity, Specificity
calculate_metrics <- function(conf_matrix) {
  precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  
  return(c(precision, sensitivity, specificity))
}

# Confusion matrix function
conf_matrix <- function(predictions, actual) {
  result <- confusionMatrix(table(predictions, actual))
  result$table
}

#(70-30 split)
conf_matrix_info_gain1 <- conf_matrix(predictions_info_gain1, testData1$diagnosis)
metrics_info_gain1 <- calculate_metrics(conf_matrix_info_gain1)

#(60-40 split)
conf_matrix_info_gain2 <- conf_matrix(predictions_info_gain2, testData2$diagnosis)
metrics_info_gain2 <- calculate_metrics(conf_matrix_info_gain2)

#(80-20 split)
conf_matrix_info_gain3 <- conf_matrix(predictions_info_gain3, testData3$diagnosis)
metrics_info_gain3 <- calculate_metrics(conf_matrix_info_gain3)

# Print accuracies and additional metrics for comparison
print("Evaluation Metrics for Information Gain with (70-30 split):")
print(paste("Accuracy:", accuracy_info_gain1))
print(paste("Precision:", metrics_info_gain1[1]))
print(paste("Sensitivity:", metrics_info_gain1[2]))
print(paste("Specificity:", metrics_info_gain1[3]))
print("Confusion Matrix:")
print(conf_matrix_info_gain1)

print("Evaluation Metrics for Information Gain with (60-40 split):")
print(paste("Accuracy:", accuracy_info_gain2))
print(paste("Precision:", metrics_info_gain2[1]))
print(paste("Sensitivity:", metrics_info_gain2[2]))
print(paste("Specificity:", metrics_info_gain2[3]))
print("Confusion Matrix:")
print(conf_matrix_info_gain2)

print("Evaluation Metrics for Information Gain with (80-20 split):")
print(paste("Accuracy:", accuracy_info_gain3))
print(paste("Precision:", metrics_info_gain3[1]))
print(paste("Sensitivity:", metrics_info_gain3[2]))
print(paste("Specificity:", metrics_info_gain3[3]))
print("Confusion Matrix:")
print(conf_matrix_info_gain3)





#Gini Index method for selecting attributes using rpart algorithm
# Load required library for gini
library(rpart)
library(rpart.plot)

# Set seed for reproducibility
set.seed(7)

# Define the formula for the decision tree
formula_gini <- diagnosis ~ .

# Fit decision tree models using Gini Index
# Build decision tree model using Gini Index selected attributes (70-30 split)
model_gini1 <- rpart(formula_gini, data = trainData1, method = "class", parms = list(split = "gini"))
# Build decision tree model using Gini Index selected attributes (60-40 split)
model_gini2 <- rpart(formula_gini, data = trainData2, method = "class", parms = list(split = "gini"))
# Build decision tree model using Gini Index selected attributes (80-20 split)
model_gini3 <- rpart(formula_gini, data = trainData3, method = "class", parms = list(split = "gini"))

# Plot the decision trees for different partition methods (Gini Index)
# Plot decision tree for (70-30 split)
rpart.plot(model_gini1, type = 2, box.palette = c("lightblue", "lightgreen"), fallen.leaves = TRUE, main = "Gini Index Decision Tree (70-30 split)")
# Plot decision tree for (60-40 split)
rpart.plot(model_gini2, type = 2, box.palette = c("lightblue", "lightgreen"), fallen.leaves = TRUE, main = "Gini Index Decision Tree (60-40 split)")
# Plot decision tree for (80-20 split)
rpart.plot(model_gini3, type = 2, box.palette = c("lightblue", "lightgreen"), fallen.leaves = TRUE, main = "Gini Index Decision Tree (80-20 split)")

# Make predictions
#(70-30 split)
predictions_gini1 <- predict(model_gini1, testData1, type = "class")
#(60-40 split)
predictions_gini2 <- predict(model_gini2, testData2, type = "class")
#(80-20 split)
predictions_gini3 <- predict(model_gini3, testData3, type = "class")

# Evaluate the models accuracy
#(70-30 split)
accuracy_gini1 <- sum(predictions_gini1 == testData1$diagnosis) / nrow(testData1)
#(60-40 split)
accuracy_gini2 <- sum(predictions_gini2 == testData2$diagnosis) / nrow(testData2)
#(80-20 split)
accuracy_gini3 <- sum(predictions_gini3 == testData3$diagnosis) / nrow(testData3)

# Function to calculate Precision, Sensitivity, Specificity
calculate_metrics <- function(conf_matrix) {
  precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  c(precision, sensitivity, specificity)
}

# Confusion matrix function
conf_matrix <- function(predictions, actual) {
  result <- confusionMatrix(table(predictions, actual))
  result$table
}

#(70-30 split)
conf_matrix_gini1 <- conf_matrix(predictions_gini1, testData1$diagnosis)
metrics_gini1 <- calculate_metrics(conf_matrix_gini1)

#(60-40 split)
conf_matrix_gini2 <- conf_matrix(predictions_gini2, testData2$diagnosis)
metrics_gini2 <- calculate_metrics(conf_matrix_gini2)

#(80-20 split)
conf_matrix_gini3 <- conf_matrix(predictions_gini3, testData3$diagnosis)
metrics_gini3 <- calculate_metrics(conf_matrix_gini3)

# Print accuracies and additional metrics for comparison
print("Evaluation Metrics for Gini Index with (70-30 split):")
print(paste("Accuracy:", accuracy_gini1))
print(paste("Precision:", metrics_gini1[1]))
print(paste("Sensitivity:", metrics_gini1[2]))
print(paste("Specificity:", metrics_gini1[3]))
print("Confusion Matrix:")
print(conf_matrix_gini1)

print("Evaluation Metrics for Gini Index with (60-40 split):")
print(paste("Accuracy:", accuracy_gini2))
print(paste("Precision:", metrics_gini2[1]))
print(paste("Sensitivity:", metrics_gini2[2]))
print(paste("Specificity:", metrics_gini2[3]))
print("Confusion Matrix:")
print(conf_matrix_gini2)

print("Evaluation Metrics for Gini Index with (80-20 split):")
print(paste("Accuracy:", accuracy_gini3))
print(paste("Precision:", metrics_gini3[1]))
print(paste("Sensitivity:", metrics_gini3[2]))
print(paste("Specificity:", metrics_gini3[3]))
print("Confusion Matrix:")
print(conf_matrix_gini3)



# Gain ratio method for selecting attributes using j48 algorithm
# Load the RWeka library
library(RWeka)

# Set seed for reproducibility
set.seed(7)

# Define the formula for the decision tree
formula_gain_ratio <- diagnosis ~ .

# Fit decision tree models using Gain Ratio
# Build decision tree model using Gain Ratio selected attributes (70-30 split)
model_gain_ratio1 <- J48(formula_gain_ratio, data = trainData1)
# Build decision tree model using Gain Ratio selected attributes (60-40 split)
model_gain_ratio2 <- J48(formula_gain_ratio, data = trainData2)
# Build decision tree model using Gain Ratio selected attributes (80-20 split)
model_gain_ratio3 <- J48(formula_gain_ratio, data = trainData3)


# Plot or print the decision trees
# (70-30 split)
plot(model_gain_ratio1, main = "Decision Tree - Gain Ratio (70-30 split)")
# (60-40 split)
plot(model_gain_ratio2, main = "Decision Tree - Gain Ratio (60-40 split)")
# (80-20 split)
plot(model_gain_ratio3, main = "Decision Tree - Gain Ratio (80-20 split)")


# Summarize the models (optional)
summary(model_gain_ratio1)
summary(model_gain_ratio2)
summary(model_gain_ratio3)

# Make predictions
# (70-30 split)
predictions_gain_ratio1 <- predict(model_gain_ratio1, testData1)
# (60-40 split)
predictions_gain_ratio2 <- predict(model_gain_ratio2, testData2)
# (80-20 split)
predictions_gain_ratio3 <- predict(model_gain_ratio3, testData3)

# Evaluate the models accuracy
# (70-30 split)
accuracy_gain_ratio1 <- sum(predictions_gain_ratio1 == testData1$diagnosis) / nrow(testData1)
# (60-40 split)
accuracy_gain_ratio2 <- sum(predictions_gain_ratio2 == testData2$diagnosis) / nrow(testData2)
# (80-20 split)
accuracy_gain_ratio3 <- sum(predictions_gain_ratio3 == testData3$diagnosis) / nrow(testData3)

# Function to calculate Precision, Sensitivity, Specificity
calculate_metrics <- function(conf_matrix) {
  precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  c(precision, sensitivity, specificity)
}

# Confusion matrix function
conf_matrix <- function(predictions, actual) {
  result <- table(predictions, actual)
  result
}

# (70-30 split)
conf_matrix_gain_ratio1 <- conf_matrix(predictions_gain_ratio1, testData1$diagnosis)
metrics_gain_ratio1 <- calculate_metrics(conf_matrix_gain_ratio1)

# (60-40 split)
conf_matrix_gain_ratio2 <- conf_matrix(predictions_gain_ratio2, testData2$diagnosis)
metrics_gain_ratio2 <- calculate_metrics(conf_matrix_gain_ratio2)

# (80-20 split)
conf_matrix_gain_ratio3 <- conf_matrix(predictions_gain_ratio3, testData3$diagnosis)
metrics_gain_ratio3 <- calculate_metrics(conf_matrix_gain_ratio3)

# Print accuracies and additional metrics for comparison
print("Evaluation Metrics for Gain Ratio with (70-30 split):")
print(paste("Accuracy:", accuracy_gain_ratio1))
print(paste("Precision:", metrics_gain_ratio1[1]))
print(paste("Sensitivity:", metrics_gain_ratio1[2]))
print(paste("Specificity:", metrics_gain_ratio1[3]))
print("Confusion Matrix:")
print(conf_matrix_gain_ratio1)

print("Evaluation Metrics for Gain Ratio with (60-40 split):")
print(paste("Accuracy:", accuracy_gain_ratio2))
print(paste("Precision:", metrics_gain_ratio2[1]))
print(paste("Sensitivity:", metrics_gain_ratio2[2]))
print(paste("Specificity:", metrics_gain_ratio2[3]))
print("Confusion Matrix:")
print(conf_matrix_gain_ratio2)

print("Evaluation Metrics for Gain Ratio with (80-20 split):")
print(paste("Accuracy:", accuracy_gain_ratio3))
print(paste("Precision:", metrics_gain_ratio3[1]))
print(paste("Sensitivity:", metrics_gain_ratio3[2]))
print(paste("Specificity:", metrics_gain_ratio3[3]))
print("Confusion Matrix:")
print(conf_matrix_gain_ratio3)





#CLUSTERING
library(factoextra)
library(ggpubr)
library(cluster)
library(NbClust)

#CLUSTERING
classLabe<- preprocessedData[,22]
#remove the class lables
unlabledData <- preprocessedData[, -22]
View(unlabledData)

unlabledData <- scale(unlabledData)

#k=2
set.seed(123)
Kresult <- kmeans(unlabledData, 2)
summary(Kresult)
fviz_cluster(Kresult, data = unlabledData)
centroid_coordinates <- Kresult$centers
print(centroid_coordinates)

###Within-cluster sum of squares 
wss <- Kresult$tot.withinss
print(wss)
# Bcubed and recall dor each clustur
cluster_assignments <- c(Kresult$cluster)
ground_truth_labels <- c(classLabe$diagnosis )

if (length(cluster_assignments) > length(ground_truth_labels)) {
  cluster_assignments <- cluster_assignments[1:length(ground_truth_labels)]
} else if (length(cluster_assignments) < length(ground_truth_labels)) {
  ground_truth_labels <- ground_truth_labels[1:length(cluster_assignments)]
}

data <- data.frame(cluster = cluster_assignments, label = ground_truth_labels)

# Function to calculate BCubed precision and recall
calculate_bcubed_metrics <- function(data) {
  n <- nrow(data)
  precision_sum <- 0
  recall_sum <- 0
  
  for (i in 1:n) {
    cluster <- data$cluster[i]
    label <- data$label[i]
    
    # Count the number of items from the same category within the same cluster
    same_category_same_cluster <- sum(data$label[data$cluster == cluster] == label)
    
    # Count the total number of items in the same cluster
    total_same_cluster <- sum(data$cluster == cluster)
    
    # Count the total number of items with the same category
    total_same_category <- sum(data$label == label)
    
    # Calculate precision and recall for the current item and add them to the sums
    precision_sum <- precision_sum + same_category_same_cluster /total_same_cluster
    recall_sum <- recall_sum + same_category_same_cluster / total_same_category
  }
  
  # Calculate average precision and recall
  precision <- precision_sum / n
  recall <- recall_sum / n
  
  return(list(precision = precision, recall = recall))
}

# Calculate BCubed precision and recall
metrics <- calculate_bcubed_metrics(data)

# Extract precision and recall from the metrics
precision <- metrics$precision
recall <- metrics$recall

# Print the results
cat("BCubed Precision:", precision, "\n")
cat("BCubed Recall:",recall,"\n")
#average silhouette for each clusters 
avg_sil <- silhouette(Kresult$cluster,dist(unlabledData)) #a dissimilarity object inheriting from class dist or coercible to one. If not specified, dmatrix must be.
fviz_silhouette(avg_sil)#k-means clustering with estimating k and initializations




#k=3
Kresult1 <- kmeans(unlabledData, 3)
summary(Kresult1)
fviz_cluster(Kresult1, data = unlabledData)
centroid_coordinates1 <- Kresult1$centers
print(centroid_coordinates1)

###Within-cluster sum of squares 
wss <- Kresult1$tot.withinss
print(wss)
# Bcubed and recall dor each clustur
cluster_assignments <- c(Kresult1$cluster)
ground_truth_labels <- c(classLabe$diagnosis )

if (length(cluster_assignments) > length(ground_truth_labels)) {
  cluster_assignments <- cluster_assignments[1:length(ground_truth_labels)]
} else if (length(cluster_assignments) < length(ground_truth_labels)) {
  ground_truth_labels <- ground_truth_labels[1:length(cluster_assignments)]
}

data <- data.frame(cluster = cluster_assignments, label = ground_truth_labels)

# Function to calculate BCubed precision and recall
calculate_bcubed_metrics <- function(data) {
  n <- nrow(data)
  precision_sum <- 0
  recall_sum <- 0
  
  for (i in 1:n) {
    cluster <- data$cluster[i]
    label <- data$label[i]
    
    # Count the number of items from the same category within the same cluster
    same_category_same_cluster <- sum(data$label[data$cluster == cluster] == label)
    
    # Count the total number of items in the same cluster
    total_same_cluster <- sum(data$cluster == cluster)
    
    # Count the total number of items with the same category
    total_same_category <- sum(data$label == label)
    
    # Calculate precision and recall for the current item and add them to the sums
    precision_sum <- precision_sum + same_category_same_cluster /total_same_cluster
    recall_sum <- recall_sum + same_category_same_cluster / total_same_category
  }
  
  # Calculate average precision and recall
  precision <- precision_sum / n
  recall <- recall_sum / n
  
  return(list(precision = precision, recall = recall))
}

# Calculate BCubed precision and recall
metrics <- calculate_bcubed_metrics(data)

# Extract precision and recall from the metrics
precision <- metrics$precision
recall <- metrics$recall

# Print the results
cat("BCubed Precision:", precision, "\n")
cat("BCubed Recall:",recall,"\n")

#average silhouette for each clusters 
avg_sil <- silhouette(Kresult1$cluster,dist(unlabledData)) #a dissimilarity object inheriting from class dist or coercible to one. If not specified, dmatrix must be.
fviz_silhouette(avg_sil)#k-means clustering with estimating k and initializations


#k=4
Kresult2 <- kmeans(unlabledData, 4)
summary(Kresult2)
fviz_cluster(Kresult2, data = unlabledData)
centroid_coordinates2 <- Kresult2$centers
print(centroid_coordinates2)

###Within-cluster sum of squares 
wss <- Kresult2$tot.withinss
print(wss)
# Bcubed and recall dor each clustur
cluster_assignments <- c(Kresult2$cluster)
ground_truth_labels <- c(classLabe$diagnosis )

if (length(cluster_assignments) > length(ground_truth_labels)) {
  cluster_assignments <- cluster_assignments[1:length(ground_truth_labels)]
} else if (length(cluster_assignments) < length(ground_truth_labels)) {
  ground_truth_labels <- ground_truth_labels[1:length(cluster_assignments)]
}

data <- data.frame(cluster = cluster_assignments, label = ground_truth_labels)

# Function to calculate BCubed precision and recall
calculate_bcubed_metrics <- function(data) {
  n <- nrow(data)
  precision_sum <- 0
  recall_sum <- 0
  
  for (i in 1:n) {
    cluster <- data$cluster[i]
    label <- data$label[i]
    
    # Count the number of items from the same category within the same cluster
    same_category_same_cluster <- sum(data$label[data$cluster == cluster] == label)
    
    # Count the total number of items in the same cluster
    total_same_cluster <- sum(data$cluster == cluster)
    
    # Count the total number of items with the same category
    total_same_category <- sum(data$label == label)
    
    # Calculate precision and recall for the current item and add them to the sums
    precision_sum <- precision_sum + same_category_same_cluster /total_same_cluster
    recall_sum <- recall_sum + same_category_same_cluster / total_same_category
  }
  
  # Calculate average precision and recall
  precision <- precision_sum / n
  recall <- recall_sum / n
  
  return(list(precision = precision, recall = recall))
}

# Calculate BCubed precision and recall
metrics <- calculate_bcubed_metrics(data)

# Extract precision and recall from the metrics
precision <- metrics$precision
recall <- metrics$recall

# Print the results
cat("BCubed Precision:", precision, "\n")
cat("BCubed Recall:",recall,"\n")
#average silhouette for each clusters 
avg_sil <- silhouette(Kresult2$cluster,dist(unlabledData)) #a dissimilarity object inheriting from class dist or coercible to one. If not specified, dmatrix must be.
fviz_silhouette(avg_sil)#k-means clustering with estimating k and initializations


fviz_nbclust(unlabledData, kmeans, method = "silhouette")+ labs(subtitle = "Silhouette method")


#optimal number of clusters is two

fres.nbclust <- NbClust(unlabledData, distance="euclidean", min.nc = 2, max.nc = 15, method="kmeans", index="all")

fviz_nbclust(unlabledData, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)+
  labs(subtitle ="Elbow method")

#Clustering center visualization
plot(unlabledData[, "compactness_mean"], col = Kresult$cluster)
points(Kresult$centers[, "compactness_mean"], col = 1:4, pch = 8, cex = 2) # plot cluster centers
