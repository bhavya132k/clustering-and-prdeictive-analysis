# Obesity Level Prediction and Clustering Analysis

## Overview

This project analyzes obesity levels in individuals from Mexico, Peru, and Colombia by studying their dietary habits, physical conditions, and lifestyle choices. The analysis is based on a dataset containing 17 attributes and 2111 records, with labels representing various obesity levels. The project employs clustering techniques and predictive models to identify the most significant factors affecting obesity and to predict obesity levels based on these attributes.

## Objectives

- Explore the relationships among different features to identify the most significant attributes for predicting obesity.
- Apply clustering methods to identify groups of individuals with similar obesity characteristics.
- Build predictive models to forecast obesity levels based on various factors, like diet, physical activity, transportation, and socio-economic conditions.
- Gain insights into the data that can inform public health policies and personalized interventions.

## Features Analyzed

- **Demographic:** Gender, Age
- **Physical Attributes:** Height, Weight
- **Dietary Habits:** Frequency of high-caloric food consumption, frequency of vegetable consumption, number of main meals
- **Lifestyle:** Physical activity frequency, time spent using technology, smoking habits
- **Others:** Family history of obesity, transportation mode, alcohol consumption

## Methodology

### Data Preprocessing
- Data normalization and transformation of categorical variables into numeric values.
- Subsetting data into training and test sets (70-30%, 60-40%, and 50-50% splits).

### Exploratory Data Analysis (EDA)
- Pairwise plotting to identify linear relationships among features.
- Correlation analysis and visualization.

### Clustering
- Applying k-means clustering to the entire dataset to identify distinct clusters.
- Determining the optimal number of clusters based on the elbow method.

### Predictive Modeling
- Building Generalized Linear Models (GLM) for different train-test splits.
- Evaluating model performance through accuracy calculations.
- Applying k-means clustering to the test set and comparing clusters with actual labels.

### Analysis and Insights
- Statistical analysis and visualization of findings.
- Analysis of the key factors influencing obesity based on predictive model outputs.

## Results and Learnings

- Identification of key features most strongly associated with obesity levels.
- Insights into clusters of individuals with similar obesity characteristics.
- Predictive models that can be used to estimate obesity levels in different demographic segments.
- Improved understanding of the factors influencing obesity for informed decision-making in public health.
