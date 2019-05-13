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
```{r}
head(citytrainframe) #I've already changed the id, time, & exit columns on excel. 
```




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

```{r}
## LIBRARIES
suppressPackageStartupMessages({ library(tidyverse) # Clean & Visual
library(data.table) })# Clean & Read
setwd("C:/Users/ssoma/Desktop/EY Challenge")
city_test <- fread(file = "keras_test3.csv")  #Using fread for faster read times as it was a large dataset
city_train <- fread(file = "keras_train3.csv")
# Creating Dataframe
citytrainframe <- data.frame(city_train, stringsAsFactors = TRUE)
citytestframe <- data.frame(city_test, stringsAsFactors = TRUE)
citytrainframe <- citytrainframe[,-c(1)]
citytestframe <- citytestframe[,-c(1)]
# Combining Train and Test Data to perform analysis on. 
kclusterdata <- rbind(citytrainframe,citytestframe)
# Seperating The date from Hash
kclusterdata <- kclusterdata %>% separate(hash, into = c("hash",  "date"), sep = "_")
kclusterdata$date <- as.numeric(kclusterdata$date)
kclusterdata$hash <- as.factor(kclusterdata$hash)
kclusterdata$hash <- as.numeric(kclusterdata$hash)
# Creating a Day column from the Date Column 
kclusterdata$day <- cut(kclusterdata$date, breaks = c(-Inf, 1,3,5,9,11,15,19,23,25,29,31), labels = c(1,3,5,2,4,1,5,2,4,1,3))
kclusterdata <- kclusterdata[,c(1,14,2:13)]
kclusterdata$day <- as.numeric(kclusterdata$day)
summary(kclusterdata)
kclusterdata
```

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

```{r}
#library(MVN)
# Visualisation
## mvn(data = kclusterdata, multivariateOutlierMethod = "quan", showOutliers = TRUE) 
## mvn(data = kclusterdata, multivariateOutlierMethod = "quan", multivariatePlot = "contour")
# Finding Mahalanobis Distance with the data with no NA values! 
MAH1 <- kclusterdata[,c(1:6)]
MAH2 <- mahalanobis(MAH1, colMeans(MAH1), cov(MAH1)) 
kclusterdata$MAH <- round(MAH2, 5)
# Assigning 1 and 0 to data which i think is valid 
kclusterdata$outlier_maha <- 0
kclusterdata$outlier_maha[kclusterdata$MAH > 10.5 ] <- 1 # 10.5 was decided via trial and error
#no_outliers <- kclusterdata[!(kclusterdata$outlier_maha== 1),]
head(kclusterdata)
```

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

```{r}
options(na.action='na.pass')
minmaxnormalise <- function(x) {(x-min(x,na.rm = TRUE))/(max(x, na.rm =TRUE)-min(x, na.rm = TRUE))}
kclusterdata$xEnNorm <- kclusterdata$x_entry %>% minmaxnormalise()
kclusterdata$yEnNorm <- kclusterdata$y_entry %>% minmaxnormalise()
kclusterdata$timeNorm <- kclusterdata$time_diff %>% minmaxnormalise()
  
head(kclusterdata)  
```



# **CLUSTERING**

## Pick the right clustering method for your data. 

**K-means Clustering:** 

**Hierarchical clustering:**

**Expectation Maximization Clustering:**

**Fuzzy clustering:** 

**Density-based clustering:**

**Model-based clustering:**

**Partitioning methods:**

[Source & read more](https://www.datanovia.com/en/blog/types-of-clustering-methods-overview-and-quick-start-r-code/)

A detailed explanation of these clustering methods are in the link above. This section of the project was by far my favourite. Clustering methods are so facinating especially when you can visualise how they work. 

Of all the clustering methods k-means and EM Clustering seem to do the best job. The visual printed below doesnt show much detail, but in the high quality saved file you can see that that the clusters have a bias towards the x-axis. Would love to know more about how others have applied clustering. 

[theme_black() credit](https://jonlefcheck.net/2013/03/11/black-theme-for-ggplot2-2/)

```{r}
# Applying Clustering to the entry points
clustertrain <- kmeans(kclusterdata[,10:11], 5000, iter.max=40)
kclusterdata$cluster <- as.numeric(clustertrain$cluster)
summary(kclusterdata)
library(gridExtra)
# Black theme 
theme_black = function(base_size = 12, base_family = "") {
 
  theme_grey(base_size = base_size, base_family = base_family) %+replace%
 
    theme(
      # Specify axis options
      axis.line = element_blank(),  
      axis.text.x = element_text(size = base_size*0.8, color = "white", lineheight = 0.9),  
      axis.text.y = element_text(size = base_size*0.8, color = "white", lineheight = 0.9),  
      axis.ticks = element_line(color = "white", size  =  0.2),  
      axis.title.x = element_text(size = base_size, color = "white", margin = margin(0, 10, 0, 0)),  
      axis.title.y = element_text(size = base_size, color = "white", angle = 90, margin = margin(0, 10, 0, 0)),  
      axis.ticks.length = unit(0.3, "lines"),   
      # Specify legend options
      legend.background = element_rect(color = NA, fill = "black"),  
      legend.key = element_rect(color = "white",  fill = "black"),  
      legend.key.size = unit(1.2, "lines"),  
      legend.key.height = NULL,  
      legend.key.width = NULL,      
      legend.text = element_text(size = base_size*0.8, color = "white"),  
      legend.title = element_text(size = base_size*0.8, face = "bold", hjust = 0, color = "white"),  
      legend.position = "right",  
      legend.text.align = NULL,  
      legend.title.align = NULL,  
      legend.direction = "vertical",  
      legend.box = NULL, 
      # Specify panel options
      panel.background = element_rect(fill = "black", color  =  NA),  
      panel.border = element_rect(fill = NA, color = "white"),  
      panel.grid.major = element_line(color = "grey35"),  
      panel.grid.minor = element_line(color = "grey20"),  
      panel.spacing = unit(0.5, "lines"),   
      # Specify facetting options
      strip.background = element_rect(fill = "grey30", color = "grey10"),  
      strip.text.x = element_text(size = base_size*0.8, color = "white"),  
      strip.text.y = element_text(size = base_size*0.8, color = "white",angle = -90),  
      # Specify plot options
      plot.background = element_rect(color = "black", fill = "black"),  
      plot.title = element_text(size = base_size*1.2, color = "white"),  
      plot.margin = unit(rep(1, 4), "lines")
 
    )
 
}
library(colorRamps)
#Visualisation of the Clusters
ggplot(kclusterdata) + theme_black() +
 aes(x_entry, y_entry, color = cluster) + 
 geom_point(alpha = 0.008, size = 0.0001) + 
 scale_color_gradientn(colours=matlab.like(5000))
  #scale_colour_viridis(option = "C") # from the viridis package
ggsave("kcluster5000black.jpg", units="in", width=14, height=10, dpi=1500)
```


# **TUNING METHOD: BAYESIAN OPTIMIZATION**
There are many tunning methods i've tried for this project.Bayesian Optimization with mlrMBO turned out to be the most efficient and accurate.   

### mlrMBO

[5.1] [Bischl B, Richter J, Bossek J, Horn D, Thomas J, Lang M (2017)._mlrMBO: A Modular Framework for Model-Based Optimization of Expensive Black-Box Functions_.](http://arxiv.org/abs/1703.03373)

[5.2] [Code sampled from Simon Coulombe](https://www.simoncoulombe.com/2019/01/bayesian/)

```{r}
library(smoof)
library(mlrMBO)
library(xgboost)
library(Matrix)
library(DiceKriging)
library(rgenoud)
# Seperating back into test and train
citytrainframe <- head(kclusterdata, n= 814262)
citytestframe <- tail(kclusterdata, n= 202937)
testans <- citytestframe[!(is.na(citytestframe$target)),] #adding test data to train data that i knew for certain would increase the accuracy 
citytrainframe <- citytrainframe[,c(1:13,15:20,14)] #rearranging  
citytestframe <- citytestframe[,c(1:13,15:20,14)]
citytrainframe <- rbind(citytrainframe, testans)
data <- (citytrainframe) 
train <- data[,1:20]
#using one hot encoding 
labels <- train$target 
new_tr <- sparse.model.matrix(target~.-1,data = train) 
#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
xgb.DMatrix.save(dtrain, 'xgb.DMatrixDtrain.data')
cv_folds = rBayesianOptimization::KFold(train$target, 
                                        nfolds= 10,
                                        stratified = TRUE,
                                        seed= 0)
# objective function: we want to maximise the log likelihood by tuning most parameters
obj.fun  <- makeSingleObjectiveFunction(
  name = "xgb_cv_bayes",
  fn =   function(x){
    set.seed(12345)
    cv <- xgb.cv(params = list(
      booster          = "gbtree",
      eta              = x["eta"],
      max_depth        = x["max_depth"],
      min_child_weight = x["min_child_weight"],
      gamma            = x["gamma"],
      subsample        = x["subsample"],
      colsample_bytree = x["colsample_bytree"],
      objective        = 'binary:logistic', 
      eval_metric     = "poisson-nloglik"),
      data = dtrain,
      nround = 30,
      folds=  cv_folds,
      prediction = FALSE,
      showsd = TRUE,
      early_stopping_rounds = 10,
      verbose = 0)
    
   cv$evaluation_log[, max(test_poisson_nloglik_mean)]
  },
  par.set = makeParamSet(
    makeNumericParam("eta",              lower = 0.001, upper = 0.05),
    makeNumericParam("gamma",            lower = 0,     upper = 5),
    makeIntegerParam("max_depth",        lower = 1,     upper = 10),
    makeIntegerParam("min_child_weight", lower = 1,     upper = 10),
    makeNumericParam("subsample",        lower = 0.2,   upper = 1),
    makeNumericParam("colsample_bytree", lower = 0.2,   upper = 1)
  ),
  minimize = FALSE
)
# generate an optimal design with only 10  points
des = generateDesign(n=10,
                     par.set = getParamSet(obj.fun), 
                     fun = lhs::randomLHS)  ## . If no design is given by the user, mlrMBO will generate a maximin Latin Hypercube Design of size 4 times the number of the black-box function’s parameters.
# i still want my favorite hyperparameters to be tested
new_params <- data.frame(max_depth = 6,
                           colsample_bytree= 0.8,
                           subsample = 0.8,
                           min_child_weight = 3,
                           eta  = 0.01,
                           gamma = 0) %>% as_tibble()
#final design  is a combination of latin hypercube optimization and my own preferred set of parameters
final_design =  new_params  %>% bind_rows(des)
# bayes will have 10 additional iterations
control = makeMBOControl()
control = setMBOControlTermination(control, iters = 10)
# run this!
run = mbo(fun = obj.fun, 
          design = final_design,  
          control = control, 
          show.info = TRUE)
write_rds( run, "run.rds")
run <- read_rds("run.rds")
# print a summary with run
run
# return  best model hyperparameters using 
run$x
# return best log likelihood using 
run$y
# return all results using 
run$opt.path$env$path
```



# **TRAINING A MODEL**

Using the best parameters from above. 

```{r}
start.time <- Sys.time()
bst_model <- xgb.train(
  data = dtrain,
  nrounds = 30000,
  objective = "binary:logistic",
  booster = "gbtree",
  eval_metric = "error",
 # watchlist = watchlist,
  nfolds = 10,
  eta = 0.001099040	,
  max.depth = 1	 ,
  min_child_weight= 4,
  gamma = 4.057689, #makes it more conservation, to avoid overfitting
  subsample = 0.3737862, #lower values prevents overfitting 
  colsample_bytree = 0.3326670	,
  missing = NA,
  seed = 333)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
```

# **EVALUATION**

```{r}
# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
#plot(e$iter, e$train_mlogloss, col = 'blue')
#lines(e$iter, e$test_mlogloss, col = 'red')
# Feature importance
imp <- xgb.importance(colnames(dtrain), model = bst_model)
print(imp)
xgb.plot.importance (importance_matrix = imp[1:8]) #top 8
```


# **SUBMISSION**

```{r}
head(citytestframe)
btestm <- sparse.model.matrix(target~.-1,citytestframe)
btest_label <- citytestframe[,"target"]
btest_matrix <- xgb.DMatrix(data = (btestm), label = btest_label) 
target <- predict(bst_model, newdata = btest_matrix)
sub1 <- read.csv("data_test.csv")
sub2 <- cbind(sub1[,c(3,11)],target)
sub3 <- sub2[(is.na(sub2$x_exit)),]
write.csv(sub3[,c(1,3)], file = "xgbpredNEW7.csv")
sub3[,c(1,3)]
```

# **APPENDIX**


## **TUNING METHOD: NESTED LOOP**

I used this method over night just to make sure that Bayesian didn't already give me the best results. This method is obviously very time consuming but useful if you know what range you're looking for. 
```{r}
data <- (citytrainframe) 
train <- data[1:20]
options(na.action='na.pass')
# binarize all factors
library(caret)
library(Metrics)
dmy <- dummyVars(" ~ .", data = train)
pTrsf <- data.frame(predict(dmy, newdata = train))
###############################################################################
# what we're trying to predict adults that make more than 50k
outcomeName <- c('target')
# list of features
predictors <- names(pTrsf)[!names(pTrsf) %in% outcomeName]
# take first 10% of the data only! Depending on your computing power
trainPortion <- floor(nrow(pTrsf)*0.1)
trainSet <- pTrsf[ 1:floor(trainPortion/2),]
testSet <- pTrsf[(floor(trainPortion/2)+1):trainPortion,]
#add eta, gamma & subsample
start.time <- Sys.time()
smallestError <- 100
for (fold in seq(6,32,1)) { #7,15,1
        for (round in seq(1,1000,1)) {  # 20,50,1
          for (etax in seq(0.001,0.25,0.001)) { #0.00,0.3, 0.02 
                # train
                bst <- xgboost(data = as.matrix(trainSet[,predictors]),
                               label = trainSet[,outcomeName],
                               max.depth=8, nround=4,
                               objective = "binary:logistic", booster = "gbtree",
                               eval_metric = "error", eta = 0.038,
                               missing = NA, # max.depth = 10,
                               min_child_weight= 8, nfold = fold,
                               gamma = 0.0472, #makes it more conservation, to avoid overfitting
                               subsample = 0.698, #lower values prevents overfitting 
                               colsample_bytree = 0.6298,
                               seed = 333, verbose=0)
                
                gc()
                
                # predict
                predictions <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
                err <- rmse(as.numeric(testSet[,outcomeName]), as.numeric(predictions))
                
                if (err < smallestError) {
                        smallestError = err
                        print(paste(fold,err))
                } 
            
          }
        }
}  
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
```


## **WIDE DATA TRAINING**

This method of turning the dataset into a wide dataset by hash proved to show no signficant improvement. However, i believe the reason for this is due to the amount of NA values in the data set.

```{r}
spreadData1 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, TENH) 
colnames(spreadData1) <- c(1:83) 
spreadData2 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, TENM) 
colnames(spreadData2) <- c(84:166) 
spreadData3 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, TEXH) 
colnames(spreadData3) <- c(167:249) 
spreadData4 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, TEXM) 
colnames(spreadData4) <- c(250:332) 
spreadData45 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, time_diff) 
colnames(spreadData45) <- c(757:840) 
spreadData5 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, x_entry) 
colnames(spreadData5) <- c(335:417) 
spreadData6 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, y_entry) 
colnames(spreadData6) <- c(419:501) 
spreadData7 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, x_exit) 
colnames(spreadData7) <- c(505:587) 
spreadData8 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, y_exit) 
colnames(spreadData8) <- c(590:672) 
spreadData9 <- kclusterdata %>% 
  group_by(hash, TENH) %>% 
  mutate(count = 1:n()) %>% 
  spread(id, cluster) 
colnames(spreadData9) <- c(674:756) 
NEWDATA <- cbind(kclusterdata[,c(1:4)] ,spreadData1[,c(19:50)], spreadData2[,c(19:50)], spreadData3[,c(19:50)], spreadData45[,c(19:50)], spreadData4[,c(19:50)], spreadData5[,c(19:50)], spreadData6[,c(19:50)], spreadData7[,c(19:50)], spreadData8[,c(19:50)], spreadData9[,c(19:50)])
NEWDATA2 <- unite(NEWDATA[,1:3], hash, sep = "_")
NEWDATA3 <- cbind(NEWDATA2, NEWDATA[,5:292]) #cahnge this
NEWDATA4 <- setDT(NEWDATA3)[, lapply(.SD, na.omit), by = hash]
NEWDATA5 <- NEWDATA4 %>% separate(hash, into = c("hash", "day", "date"), sep = "_")
target <- fread(file = "target.csv") #change the 0time diff to 1 and 0 maybe?
NEWDATA6 <- cbind(NEWDATA5, target) 
BIG_LETTERS <- c(LETTERS,
                 do.call("paste0",CJ(LETTERS,LETTERS)),
                 do.call("paste0",CJ(LETTERS,LETTERS,LETTERS)))
colnames(NEWDATA6) <- c("hash", "day", "date", BIG_LETTERS[c(1:288)] , "target") #change this
NEWDATA6$hash <- as.numeric(NEWDATA6$hash)
NEWDATA6$day <- as.numeric(NEWDATA6$day)
NEWDATA6$date <- as.numeric(NEWDATA6$date)
head(NEWDATA6, n = 20)
```
