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