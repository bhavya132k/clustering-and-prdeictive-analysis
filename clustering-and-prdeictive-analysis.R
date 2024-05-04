# Load necessary libraries
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr) 
library(caret) 
library(cluster) 
library(corrplot)
library(glmnet)
library(rstatix)
library(gmodels)
library(psych)
library(nnet)

# Load the Obesity Dataset
obesityData <- read.csv("/Users/bhavya/Academics/ThirdSem/BDA/Project2/ObesityDataSet_raw_and_data_sinthetic.csv")
#obesityData <- read_csv("Desktop/BDA/Project2/obesity.csv")

# Task 1: Data Exploration through Pairwise Plotting

# Inspect the structure and head of the dataset
str(obesityData)
head(obesityData)

# Convert non-numeric columns to numeric where necessary
non_numeric_columns <- sapply(obesityData, Negate(is.numeric))
obesityData[non_numeric_columns] <- lapply(obesityData[non_numeric_columns], function(x) as.numeric(as.factor(x)))
obesityData

# Pairwise plotting to identify relationships
pairs(obesityData[,1:16])  # Adjust based on your columns of interest

# After conversion, check the structure again
str(obesityData)
head(obesityData)
summary(obesityData)
describe(obesityData)

# Task 2: Prepare the Data

# Normalize features, keeping the response variable "NObeyesdad" intact
features <- setdiff(names(obesityData), "NObeyesdad")
obesityData.features <- obesityData[features]
normalize <- function(x) {((x-min(x))/(max(x)-min(x)))}
obesityData.features.norm <- as.data.frame(lapply(obesityData.features, normalize))
obesityData.norm <- cbind(obesityData.features.norm, NObeyesdad = obesityData$NObeyesdad)

# Correlation analysis on the normalized features
correlation <- cor(obesityData.features.norm)

# Plot the correlation matrix
corrplot(correlation, method = "circle")

# Split the dataset into Training and Test Sets (70-30%, 60-40%, 50-50%)
set.seed(123)  # Ensures reproducibility
# For 70-30 Split
split_ratio_70_30 <- 0.7
obesityData_norm_rows_70_30 <- nrow(obesityData.norm)
obesityData_rows_70_30 <- round(split_ratio_70_30 * obesityData_norm_rows_70_30)
obesityData_train_index_70_30 <- sample(obesityData_norm_rows_70_30, obesityData_rows_70_30)
obesityData_train_70_30 <- obesityData.norm[obesityData_train_index_70_30, ]
obesityData_test_70_30 <- obesityData.norm[-obesityData_train_index_70_30, ]
cat("70-30 Split: Training set has", nrow(obesityData_train_70_30), "rows. Test set has", nrow(obesityData_test_70_30), "rows.\n")

# For 60-40 Split
split_ratio_60_40 <- 0.6
obesityData_norm_rows_60_40 <- nrow(obesityData.norm)
obesityData_rows_60_40 <- round(split_ratio_60_40 * obesityData_norm_rows_60_40)
obesityData_train_index_60_40 <- sample(obesityData_norm_rows_60_40, obesityData_rows_60_40)
obesityData_train_60_40 <- obesityData.norm[obesityData_train_index_60_40, ]
obesityData_test_60_40 <- obesityData.norm[-obesityData_train_index_60_40, ]
cat("60-40 Split: Training set has", nrow(obesityData_train_60_40), "rows. Test set has", nrow(obesityData_test_60_40), "rows.\n")

# For 50-50 Split
split_ratio_50_50 <- 0.5
obesityData_norm_rows_50_50 <- nrow(obesityData.norm)
obesityData_rows_50_50 <- round(split_ratio_50_50 * obesityData_norm_rows_50_50)
obesityData_train_index_50_50 <- sample(obesityData_norm_rows_50_50, obesityData_rows_50_50)
obesityData_train_50_50 <- obesityData.norm[obesityData_train_index_50_50, ]
obesityData_test_50_50 <- obesityData.norm[-obesityData_train_index_50_50, ]
cat("50-50 Split: Training set has", nrow(obesityData_train_50_50), "rows. Test set has", nrow(obesityData_test_50_50), "rows.\n")



# Task 3: Clustering on the Whole Data Set

#Creating 5 clusters
obesity.k5 <- kmeans(obesityData.norm, centers = 5)
str(obesity.k5)
obesity.k5
factoextra::fviz_cluster(obesity.k5,obesityData.norm)
obesityData[194,]
obesityData[578,]

#Creating 7 clusters
obesity.k7 <- kmeans(obesityData.norm, centers = 7)
str(obesity.k7)
obesity.k7
factoextra::fviz_cluster(obesity.k7,obesityData.norm)

#Creating 9 clusters
obesity.k9 <- kmeans(obesityData.norm, centers = 9)
str(obesity.k9)
obesity.k9
factoextra::fviz_cluster(obesity.k9,obesityData.norm)

#Creating 11 clusters
obesity.k11 <- kmeans(obesityData.norm, centers = 11)
obesity.k11
str(obesity.k11)
factoextra::fviz_cluster(obesity.k11,obesityData.norm)

#Creating 15 clusters
obesity.k15 <- kmeans(obesityData.norm, centers = 15)
str(obesity.k15)
obesity.k15
factoextra::fviz_cluster(obesity.k15,obesityData.norm)
obesityData[269,]
obesityData[164,]

#Finding the optimal k value
factoextra::fviz_nbclust(obesityData,FUNcluster = kmeans,method = "wss",k.max=20,verbose = TRUE)


# Task 4: Prediction with Training and Test Sets

# GLM for 70-30% Split
obesityData_train_glm_70_30 <- glm(NObeyesdad ~ ., data = obesityData_train_70_30, family = "gaussian")
#obesityData_train_glm_70_30 <- glm(formula = obesityData.train$NObeyesdad ~ obesityData.train$Age+obesityData.train$Height+obesityData.train$Weight+obesityData.train$MTRANS,family = gaussian, data=obesityData.train)
obesityData_test_pred_70_30 <- predict(obesityData_train_glm_70_30, newdata = obesityData_test_70_30, type = "response")

# Adjusting predictions and generating true labels
obesityData_test_pred_class_70_30 <- ifelse(obesityData_test_pred_70_30 > 0.5, 1, 0)  # Adjust according to your outcome
true_labels_70_30 <- obesityData_test_70_30$NObeyesdad

# Accuracy Calculation
accuracy_70_30 <- mean(true_labels_70_30 == obesityData_test_pred_class_70_30)
cat("Accuracy for GLM predictions (70-30 split):", accuracy_70_30, "\n")

# K-Means Clustering for the Test Set (70-30% Split)
k_clusters_70_30 <- 6  # Change based on your needs
predicted_clusters_70_30 <- kmeans(obesityData_test_70_30[, -ncol(obesityData_test_70_30)], centers = k_clusters_70_30)$cluster

# Cross-Tabulation to Compare Clusters and True Labels
obesityData_test_ct_k6_70_30 <- CrossTable(as.factor(true_labels_70_30), as.factor(predicted_clusters_70_30), prop.chisq = FALSE)

# Anova Test for GLM Model
obesityData_train_glm_anova_70_30 <- anova(obesityData_train_glm_70_30, test = "Chisq")
plot(obesityData_train_glm_70_30)
confint(obesityData_train_glm_70_30)

# Summarizing GLM Model
summary(obesityData_train_glm_70_30)


# GLM for 60-40% Split
obesityData_train_glm_60_40 <- glm(NObeyesdad ~ ., data = obesityData_train_60_40, family = "gaussian")
obesityData_test_pred_60_40 <- predict(obesityData_train_glm_60_40, newdata = obesityData_test_60_40, type = "response")

# Adjusting predictions and generating true labels for 60-40 split
obesityData_test_pred_class_60_40 <- ifelse(obesityData_test_pred_60_40 > 0.5, 1, 0)
true_labels_60_40 <- obesityData_test_60_40$NObeyesdad

# Accuracy Calculation for 60-40 split
accuracy_60_40 <- mean(true_labels_60_40 == obesityData_test_pred_class_60_40)
cat("Accuracy for GLM predictions (60-40 split):", accuracy_60_40, "\n")

# K-Means Clustering for the Test Set (60-40% Split)
k_clusters_60_40 <- 6  # Adjust based on your needs
predicted_clusters_60_40 <- kmeans(obesityData_test_60_40[, -ncol(obesityData_test_60_40)], centers = k_clusters_60_40)$cluster

# Cross-Tabulation to Compare Clusters and True Labels for 60-40 split
obesityData_test_ct_k6_60_40 <- CrossTable(x = as.factor(true_labels_60_40), y = as.factor(predicted_clusters_60_40), prop.chisq = FALSE)

# Anova Test for GLM Model for 60-40 split
obesityData_train_glm_anova_60_40 <- anova(obesityData_train_glm_60_40, test = "Chisq")
plot(obesityData_train_glm_60_40)

# Summarizing GLM Model for 60-40 split
summary(obesityData_train_glm_60_40)


# GLM for 50-50% Split
obesityData_train_glm_50_50 <- glm(NObeyesdad ~ ., data = obesityData_train_50_50, family = "gaussian")
obesityData_test_pred_50_50 <- predict(obesityData_train_glm_50_50, newdata = obesityData_test_50_50, type = "response")

# Adjusting predictions and generating true labels for 50-50 split
obesityData_test_pred_class_50_50 <- ifelse(obesityData_test_pred_50_50 > 0.5, 1, 0)
true_labels_50_50 <- obesityData_test_50_50$NObeyesdad

# Accuracy Calculation for 50-50 split
accuracy_50_50 <- mean(true_labels_50_50 == obesityData_test_pred_class_50_50)
cat("Accuracy for GLM predictions (50-50 split):", accuracy_50_50, "\n")

# K-Means Clustering for the Test Set (50-50% Split)
k_clusters_50_50 <- 6  # Adjust based on your needs
predicted_clusters_50_50 <- kmeans(obesityData_test_50_50[, -ncol(obesityData_test_50_50)], centers = k_clusters_50_50)$cluster

# Cross-Tabulation to Compare Clusters and True Labels for 50-50 split
obesityData_test_ct_k6_50_50 <- CrossTable(x = as.factor(true_labels_50_50), y = as.factor(predicted_clusters_50_50), prop.chisq = FALSE)

# Anova Test for GLM Model for 50-50 split
obesityData_train_glm_anova_50_50 <- anova(obesityData_train_glm_50_50, test = "Chisq")
plot(obesityData_train_glm_50_50)

# Summarizing GLM Model for 50-50 split
summary(obesityData_train_glm_50_50)






