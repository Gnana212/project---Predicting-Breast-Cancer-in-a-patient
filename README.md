# project---Predicting-Breast-Cancer-in-a-patient- Overview

Implementation of ensembling techniques(Supervised-learning) to perform classification on the dataset of Breast Cancer to predict if the patient has breast cancer or not.

In this project, first I have analysed the  data and explored the relationships among  variables. Next, I am going to pre-process the data .After that I trained SVM classifier to predict whether the patient has cancer or not  and then  assess correctness in classifying data  to efficiency and effectiveness of the SVM classifier in terms of accuracy, precision, sensitivity, specificity and AUC ROC and then tuning  hyperparameters of SVM Classifier  by  scikit-learn library. Here I have used Machine Learning Algorithms - Logistic Regression, Gradient Boosting Classifier, Random Forest Classifier, Decision Tree Classifier, Kneighbours Classifier, XGB Classifier, Supportr vector Classifier


Dataset Information:

 * The dataset consists of several predictor variables and one target variable, Diagnosis.
 *  The target variable has values 'Benign' and 'Malignant', where 'Benign' means that the cells are not harmful or there is no cancer and 'Malignant' means that the patient has cancer and the cells have a harmful effect 
 
 Input Variable:
 
radius              ->  Mean of distances from center to points on the perimeter 
texture             -> Standard deviation of gray-scale values 
perimeter           -> Observed perimeter of the lump 
area                -> Observed area of lump 
smoothness          -> Local variation in radius lengths 
compactness         -> perimeter^2 / area - 1.0 
concavity           -> Severity of concave portions of the contour 
concave points      -> number of concave portions of the contour 
symmetry            -> Lump symmetry
fractal dimension   -> "coastline approximation" - 1
Diagnosis           -> Whether the patient has cancer or not? ('Malignant','Benign')


Code and Resources Used:

Packages : Pandas, Numpy, Matplotlib, Seaborn, Sklearn.



Model Building:

By using the Accuracy_Score:
                * Logistic Regression
                * Random Forest Classifier
                * Decision Tree Classifier
                * Kneighbours Classifier
                * Supportr vector Classifier
                * Gradient Boosting Classifier
                * XGB Classifier
                 
                 
                 
Model Performance:


    * Accuracy_Score
    * Recall
    * Precision
    * Classification Report
    * The ROC Curve
                 
                 


