# University of Windsor - COMP 4740 - Machine Learning

## Machine Learning for Prostate Cancer Detection

### Requirements
•	Python 3.7  
•	Numpy (Numpy arrays)  
•	Pandas (Dataframes)  
•	Scikit Learn 0.24.1 (Machine Learning Models)  
•	Imbalanced-Learn 0.8.0 (SMOTE / ENN)  

### Dataset
•	Prostate Adenocarcinoma (TCGA, Firehose Legacy) via cBioPortal.org.  
•	A compressed version of the .csv file is included in the repository.  
•	The data set contains RNA expression values for roughly 60 000 genes for 494 patients who have undergone biopsies for prostate cancer.  
•	Also provided are the Gleason Scores, as well as several other metrics used in determining cancer risk, for each of the patients.  

### Purpose
•	Our goal was to construct a model that correctly classified patients as belonging to one of five categories based on their Gleason Score (6,7,8,9,10).  
•	We applied machine learning techniques for data processing, feature selection, class rebalancing, and classification in order to maximize
classification accuracy.  
•	All models (Naive Bayes, SVM, KNN, Random Forest) eventually achieved high (95% +) accuracy - both with and without dimensionality reduction
via Linear Discriminant Analysis.  

### Instructions for Use
•	All libraries listed above should be installed and working.  
•	Clone the repository.  
•	Unzip the .csv file and ensure that it remains in the working directory with main.py and pipepline.py.  
•	Run the program via main.py - the process should complete in under 3 minutes.  
NOTE: main.py is the driver program, while all functions are stored in pipeline.py.  
NOTE: Some IDEs may be unable to load the dataset due to its size - we had success with PyCharm and Spyder.  
NOTE: We performed all of our tests on a Windows operating system.  

### TO-DO
•	There are still opportunities for further optimization via additional / different feature selection and classification methods.  
•	We would like to add additional data visualization methods.  
•	We would like to perform additional tests with different target variables (other than Gleason Score).  
