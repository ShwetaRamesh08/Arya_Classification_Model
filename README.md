# Problem Statement
Binary Classification - Do an exploratory analysis of the dataset provided, decide on feature selection, preprocessing before training a model to classify as class ‘0’ or class ‘1’.

## Datasets Given : 
1.	training_set.csv - To be used as training and validation set - 3910 records, 57 features, 1 output
2.	test_set.csv (without Ground Truth) - 691 records, 57 features


# Approach: 

1. **EDA/Data Preprocessing:**

    * As part of the EDA and Data Preprocessing, we first look at the datasets given to us and check if there are any additional columns that we wouldn't need for building/testing the model. We find that there's an Unnamed column in both the train as well as the test data, which just contains the row numbers. Since that doesn't add any new information, we drop the column.
    
    * We then look for missing values in both the datasets. We find that there are no missing values.
    
    * We check if the given training dataset is balanced. To do this, we do a value count of 0s and 1s in the target variable. We find that the proportion is aroynd 60-40, hence the dataset seems to be balanced.
    
    * We check the distribution of each feature using histograms. We find that the distribution of each feature is heavily skewed. We try to transform the features to Normality by using the **YeoJohnson transformation**. Since a number of observations are 0, Box-Cox transformation fails to work.
    
    * We also check the class-wise distribution of the features to check if any pattern is being observed. Post that, we create a correlation matrix to see if the features are correlated to each other. We find that there is some correlation between the variables. To make sure that this doesn't affect our model, we perform feature selection.
    
    
2. **Feature Selection:**
    
    * We check for multicollinearity using the Variance Inflation Factor (VIF). A feature with VIF value > 5 suggests that it can be expressed as a function of the other features, and hence doesn't provide much information on its own to the model. Hence the features with VIF > 5 are dropped one by one. 
    
    * We then create a benchmark Logistic regression model consisting of all the features except the ones dropped in the previous step, to check the Wald-p values. The variables with a p-value > 0.05 are considered to be insignificant and hence are dropped. The dataset is first split into a train and validation set in the ratio 4:1.
    
    * While creating the benchmark logistic model, it we find that some separation exists in the data. Thus the maximum likelihood estimate did not exist for that model. 
    
    * We then apply penalized likelihood regression technique named the Firth Regression Technique. Since this technique does not exist in either the statsmodels or the sklearn libraries, John Lee's implementation of the same (link : https://gist.github.com/johnlees/3e06380965f367e4894ea20fbae2b90d) is used, with minor modifications.
    
    * We further drop the features with p-values > 0.05, and come up with a list of significant features that can be used to build the model.
    
    
3. **Model building and Validation:**
    
    * We use the Firth Regression to identify the coefficients of the features and then use those coefficients to predict the Y values using the logistic function. 
    
    * We use a default threshold of 0.5 to classify the predicted values as either 0 or 1.
    
    * We then analyse the performance of the model based on various metrics (using the confusion matrix). We consider the following 
         
        * TP (True Positive): Actual value = 1, predicted value = 1
        * TN (True negative): Actual value = 0, predicted value = 0
        * FP (False Positive): Actual value = 0, predicted value = 1
        * FN (False Negative): Actual value = 1, predicted value = 0
        
   * We use the following metrics to test the performance:
        1. Accuracy: Measures the proportion of correct predictions to the total number of predictions
        
           * formula : (TP + TN)/(TP + TN + FP + FN)
        2. Sensitivity: The ability of the model to correctly identify a positive. Also known as Recall.
           * formula : (TP)/(TP + FN)
        3. Specificity: The ability of the model to correctly identify a negative.
           * formula : (TN)/(TN + FP)
        4. Precision: Proportion of true positives to the total number of positives predicted
           * formula : (TP)/(TP + FP)
        5. F1-Score: The harmonic mean of Prediction and Recall (Sensitivity). The F1-score is high when there is a balance between Precision and Recall.
           * formula : 2xPrecisionxRecall/(Precision + Recall)


4. **Predictions on the test set:**

    * We first select only those features that are required for predicting the model.
    * We then transform the data to Normality using the YeoJohnson transformation. 
    * We then make predictions based on the coefficients determined earlier.

# Files in this project:
 Along with this readme file, this project also contains the following files:
 1. EDA_Data_Preprocessing.ipynb : A python Notebook containing the approach and codes used for EDA and Feature Selection.
 2. FirthRegression.py : A script for the Firth Regression Algorithm based on the code implemented by John Lee.
 3. model_performance.ipynb : A python Notebook containing the approach and codes used for testing the model on the validation set and making predictions on the test set.
 4. training_set.csv : The csv for the train set as provided in the problem statement.
 5. test_set.csv: The csv for the test set as provided in the problem statement.
 6. modified_train_set.csv : The train set generated after data preprocessing, feature selection, and the train-test split.
 7. Validation_set.csv : The validation set generated after data preprocessing and the train-test split.
 8. predictions.csv : The original test set, with an additional column of predicted Y.
 9. requirements.txt : A file containing the list of libraries/dependencies along with their versions, used to run this code.
