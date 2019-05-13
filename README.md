GOTTA ADD IN AREAS OF IMPROVEMENT, 
get rid of code and add pictures
show screenshots of the data and etc

---
title: "Optimizing Data for XGBOOST"
Author: Harri Somakanthan
output: html_notebook
---

This is both a personal guide & my approach to the EY Data Science Challenge 2019:

* Pre-processing data
* Dectecting outliers & cleaning data
* Importance of normalisation & balancing in datasets
* Clustering methods
* Tunning using bayesian optimization & xgb.cv 
* Training a model
* Evalutation & correlations
* Submissions 

***

Context: 
Students will have access to a data file data_train.csv that contains the anonymized geolocation data of multiple mobile devices in the City of Atlanta (US) for 11 working days in October 2018. The devices’ ID resets every 24 hours; therefore, you will not be able to trace the same device across different days. Therefore, every device ID represents a 1-day journey. Each journey is formed by several trajectories. A trajectory is defined as the route of a moving person in a straight line with an entry and an exit point.


# **PRE-PROCESSING**

### What hidden data can we extract?
* After quick look of the data set provided on excell, i noticed that each hash *"0467278736f0b4abf9ea7fc75ae634ac_29"* ends in the date. Hence i was able to create two new additional pieces of data; Date & Day of the week. 
* I also noticed that each trajectory *"traj_0467278736f0b4abf9ea7fc75ae634ac_29_8"* ended in the number of the trajectory and was able to seperate this into a numeric only column.
* I seperated time *"8:16:37"* into hour, minutes & time_diff
* I decided to remove the velocity columns as they had too many NA values and would be of no help. 
* Distance can be calculated from the two (x1,y1) & (x2,y2) coordinates, which also means velocity can be calculated. You will noticed that i removed this data later. I will explain towards the training section. 
* Lastly i removed the x_exit & y_exit of every trajectory at hour 15 & 16 so that both the test and train sets are void of bias. BALANCE! 

### Can you add more data?
* YES! In traffic engineering, roads use something called a k-factor to determine traffic flow. Luckily Georgia Atlanta has this informattion publically avaliable on their GDOT API, however, our data does not have longitudes & latitudes which means unfortunetly i could not sync this data. 
* Weather data could have also been added, however, there was no snow in the month of October 2018 and the rain was less than 0.1mm. 
* Possibly by turning the data into a widedata set by hash! 

# **OUTLIERS**

### Univariate Outlier Detection Methods:

 * **Tukey’s method of outlier detection:**
Tukey’s rule says that the outliers are values more than 1.5 times the interquartile range from the quartiles — either below Q1 − 1.5IQR, or above Q3 + 1.5IQR. 

 * **z-score:** Using the outliers package.
"z" calculates normal scores (differences between each value and the mean divided by sd).

### Multivariate Outlier Detection Methods:

* **Bivariate box plots and scatter plots:** same as above but with more variables.

* **The Mahalanobis distance:** Using the MVN package. 

Using trail and error i applied all the above to various aspects of the data to determine outliers. I found Mahalanobis distance to give the best results. 


### Handling Outliers:

**Excluding:** df[is.na(df),] <- NULL 

**Imputing:** Look up guides on mice, missForest, impute, missRanger. This approach may not always be best, sometimes its best to leave values as Na's. Do a test of imputed data vs non-imputed to see what works for you. 

**Capping:** NA

In this project because i assigned 1 and 0s to the Maha values therefore i had no need to remove any data. I did however, attempt to use missRanger and Mice to impute the values in a seperate trial where i combined the orginial train and test data and imputed the NA values in test but, the results were poor. 

### Feature extraction:

**Principal Component Analysis (PCA):** prcomp, caret, mlr are useful packages for this.
preProcess(df, method = "pca", thresh = 0.90) -- cumulative explained variance of 90%. 
preProcess(df, method = "pca", pcaComp = *number_of_dimensions_if_known*) 

I did attempt some PCA, however, this showed no improvement in results and hence didn't end up using it. 


# **NORMALISATION**

### Pick the right nromalisation method for your data. 

**Centering and scaling:** use the scale() function. For mean centering use (center = TRUE, scale = FALSE). For scaling (center = FALSE, scale = TRUE). For RMS (center = FALSE, scale = apply(df, 2, sd, na.rm = TRUE))  

**z score standardisation:**  using the scale() with the center = TRUE, scale = TRUE  

**Min- Max Normalisation:** Feel free to use this in combination with lapply() or pipes. 
minmaxnormalise <- function(x) {(x-min(x,na.rm = TRUE))/(max(x, na.rm =TRUE)-min(x, na.rm = TRUE))}

### Data transformation for skewed data. 

**Box-Cox Transformation:** The caret, MASS, forecast, geoR, EnvStats, and AIS packages can all perform BCT. I personally like Forecast 
as it has functions to find the best lamba parameter.  

**Data Transformation via Mathematical Operations:** Use log10(), log()  

### Binning/ Discretisation

**Equal width (distance) binning:** Use the infotheo package and the discretize() function

**Equal depth (frequency) binning:**  discretize() function with disc = "equalfreq"


I used all the above methods when it came to normalising the data thorugh trial and error. I found that min-max, centering and scaling were the best options. 


# **CLUSTERING**

## Pick the right clustering method for your data. 

**K-means Clustering, Hierarchical clustering, Expectation Maximization Clustering, Fuzzy clustering, Density-based clustering
Model-based clustering or Partitioning methods**

[Source & read more](https://www.datanovia.com/en/blog/types-of-clustering-methods-overview-and-quick-start-r-code/)

A detailed explanation of these clustering methods are in the link above. This section of the project was by far my favourite. Clustering methods are so facinating especially when you can visualise how they work. 

Of all the clustering methods k-means and EM Clustering seem to do the best job. The visual printed below doesnt show much detail, but in the high quality saved file you can see that that the clusters have a bias towards the x-axis. Would love to know more about how others have applied clustering. 

[theme_black() credit](https://jonlefcheck.net/2013/03/11/black-theme-for-ggplot2-2/)

![kcluster5000black.jpg](./kcluster5000black.jpg)


# **TUNING METHOD: BAYESIAN OPTIMIZATION**
There are many tunning methods i've tried for this project.Bayesian Optimization with mlrMBO turned out to be the most efficient and accurate.   

### mlrMBO

[5.1] [Bischl B, Richter J, Bossek J, Horn D, Thomas J, Lang M (2017)._mlrMBO: A Modular Framework for Model-Based Optimization of Expensive Black-Box Functions_.](http://arxiv.org/abs/1703.03373)

[5.2] [Code sampled from Simon Coulombe](https://www.simoncoulombe.com/2019/01/bayesian/)


# **TRAINING A MODEL**

Using the best parameters from above. 

# **EVALUATION**



# **SUBMISSION**


# **APPENDIX**


## **TUNING METHOD: NESTED LOOP**

I used this method over night just to make sure that Bayesian didn't already give me the best results. This method is obviously very time consuming but useful if you know what range you're looking for. 


## **WIDE DATA TRAINING**

This method of turning the dataset into a wide dataset by hash proved to show no signficant improvement. However, i believe the reason for this is due to the amount of NA values in the data set.
