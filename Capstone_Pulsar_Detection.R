# Download the required packages from CRAN and install
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(ROCR)) install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(boot)) install.packages("boot", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

# Download the HTRU2 dataset from UCI website and store it to a temporary variable
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip", dl)

# Extract the HTRU_2.csv file from the zip file and load it to HTRU object. Apply suitable column names
# The confirmed column has the classes 1 or 0 indicating whether the signal is pulsar of not.
HTRU <- fread(text = gsub(",", "\t", readLines(unzip(dl, "HTRU_2.csv"))), 
                col.names = c("mean_ip","sd_ip","excess_kurt_ip","skew_ip","mean_dmsnr",
                              "sd_dmsnr","excess_kurt_dmsnr","skew_dmsnr","confirmed"))

# The confirmed column has values 0 and 1. Yet the column is of class numeric. Change the column to factors.
HTRU$confirmed <- as.factor(HTRU$confirmed)

# Split the HTRU dataset in train and validation datasets in the ratio 90:10. Validation dataset will be used in the end to 
# make prediction based on the final model. The train dataset will be used for model training and cross validation.
set.seed(1, sample.kind = "Rounding")
test.index <- createDataPartition(y = HTRU$confirmed, times = 1, p = 0.1, list = FALSE)
HTRU.train <- HTRU[-test.index,]
HTRU.validation <- HTRU[test.index,]

# Split the train dataset further into train and test datasets for model training and cross validation.
set.seed(1, sample.kind = "Rounding")
train.index <- createDataPartition(y = HTRU.train$confirmed, times = 1, p = 0.1, list = FALSE)
HTRU.train.train <- HTRU.train[-train.index,]
HTRU.train.test <- HTRU.train[train.index,]

# Fit a logistic regression model first to verify its performance. The prediction result of logistic regression
# is the probability of a candidate being a pulsar.
glm.train.fit <- glm(confirmed ~ ., data = HTRU.train.train, family = "binomial")
p_hat.train.glm <- predict(glm.train.fit, newdata = HTRU.train.test, type = "response")

# With a default cutoff of 0.5 make prediction on test set and create the confusion matrix.
y_hat.train.glm <- ifelse(p_hat.train.glm > 0.5, 1, 0) %>% factor
confusionMatrix(y_hat.train.glm,HTRU.train.test$confirmed, positive = "1")

# Verify if cross validation can impprove the model performance
# Perform cross-validation to find out the ideal cutoff based on the highest f-score. 
cutoff_seq.glm <- seq(0, 1, by = 0.05)
p_hat.glm.df <- data.frame()
for(p in cutoff_seq.glm){
  y_hat.train.glm <- ifelse(p_hat.train.glm > p, 1, 0) %>% factor
  conf_mat <- confusionMatrix(y_hat.train.glm,HTRU.train.test$confirmed, positive = "1")
  df_row <- data.frame(cutoff = p, Sens = conf_mat$byClass["Sensitivity"], Spec = conf_mat$byClass["Specificity"], 
                       fscore = 2* ((conf_mat$byClass["Sensitivity"]*conf_mat$byClass["Specificity"])/
                                      (conf_mat$byClass["Sensitivity"]+conf_mat$byClass["Specificity"])))
  p_hat.glm.df <- rbind(p_hat.glm.df,df_row)
}
cutoff.glm <- p_hat.glm.df[which(p_hat.glm.df$fscore == max(p_hat.glm.df$fscore)),]$cutoff

# Use the cutoff value derived from cross-validation to make prediction on the test set once again
y_hat.train.glm <- ifelse(p_hat.train.glm > cutoff.glm, 1, 0) %>% factor
conf_mat <- confusionMatrix(y_hat.train.glm,HTRU.train.test$confirmed, positive = "1")

# Get the important statistics of the testing to eventually compare and finalize the model.
pred <- prediction(as.numeric(p_hat.train.glm), HTRU.train.test$confirmed)
auc.glm <- as.numeric(performance(pred,  measure  = "auc")@y.values)
model.results <- data.frame(Model = "glm", CutOff = cutoff.glm, sensitivity = conf_mat$byClass["Sensitivity"], 
                            Acc = conf_mat$overall["Accuracy"], PPV = conf_mat$byClass["Pos Pred Value"],
                            AUC = auc.glm)

# Try boosting algorithm to see if performance can be improved upon glm
# Train the training model on xgboost algorithm
HTRU.train.train$confirmed <- ifelse(HTRU.train.train$confirmed == 1,1,0) # Convert factors to numeric for xgboost
xgb.train.data <- model.matrix(~., data = HTRU.train.train[,-c("confirmed")])
xgb.train.labels <- HTRU.train.train$confirmed
xgb.test.data <- model.matrix(~., data = HTRU.train.test[,-c("confirmed")])
xgb.test.labels <- HTRU.train.test$confirmed

xgb.train <- xgb.DMatrix(data = xgb.train.data, label = xgb.train.labels)
xgb.test <- xgb.DMatrix(data = xgb.test.data, label = xgb.test.labels)

xgboost.train.fit <- xgb.train(data= xgb.train, booster = "gbtree", nrounds = 75, objective = "binary:logistic", 
                         eta=0.3, gamma=0, max_depth=7, min_child_weight=1, subsample=1, colsample_bytree=1)
p_hat.train.xgb <- predict(xgboost.train.fit, newdata = xgb.test, type = "response")

# With a default cutoff of 0.5 the prediction on test set was perforrmed and the confusion matrix was created.
y_hat.train.xgb<- ifelse(p_hat.train.xgb > 0.50, 1, 0) %>% factor
conf_mat <- confusionMatrix(y_hat.train.xgb,HTRU.train.test$confirmed, positive = "1")

# Similar to glm, find the cutoff with the highest f-score.
cutoff_seq.xgb <- seq(0, 1, by = 0.05)
p_hat.xgb.df <- data.frame()
for(p in cutoff_seq.xgb){
  y_hat.train.xgb <- ifelse(p_hat.train.xgb > p, 1, 0) %>% factor
  conf_mat <- confusionMatrix(y_hat.train.xgb,HTRU.train.test$confirmed, positive = "1")
  df_row <- data.frame(cutoff = p, Sens = conf_mat$byClass["Sensitivity"], Spec = conf_mat$byClass["Specificity"], 
                       fscore = 2* ((conf_mat$byClass["Sensitivity"]*conf_mat$byClass["Specificity"])/
                                      (conf_mat$byClass["Sensitivity"]+conf_mat$byClass["Specificity"])))
  p_hat.xgb.df <- rbind(p_hat.xgb.df,df_row)
}
cutoff.xgb <- p_hat.xgb.df[which(p_hat.xgb.df$fscore == max(p_hat.xgb.df$fscore)),]$cutoff

# With the optimal cutoff derived from the above calculation make the predition on the test dataset once again.
y_hat.train.xgb<- ifelse(p_hat.train.xgb > cutoff.xgb, 1, 0) %>% factor
conf_mat <- confusionMatrix(y_hat.train.xgb,HTRU.train.test$confirmed, positive = "1")

# Get the important statistics of the testing to eventually compare and finalize the model.
pred <- prediction(as.numeric(p_hat.train.xgb), HTRU.train.test$confirmed)
auc.xgb <- as.numeric(performance(pred,  measure  = "auc")@y.values)
model.results <- bind_rows(model.results, data.frame(Model = "xgb", CutOff = cutoff.xgb, 
                                                     sensitivity = conf_mat$byClass["Sensitivity"], 
                                                     Acc = conf_mat$overall["Accuracy"], 
                                                     PPV = conf_mat$byClass["Pos Pred Value"],
                                                     AUC = auc.xgb))

model.results %>% knitr::kable()

# Prediction on validation dataset
# While the accuracy, sensitivity and AUC values for both the models are almost similar, the PPV of the boosting
# model is higher than the logistic regression model. Hence, it was chosen as the final model.
HTRU.train$confirmed <- ifelse(HTRU.train$confirmed == 1,1,0)
xgb.train.data <- model.matrix(~., data = HTRU.train[,-c("confirmed")])
xgb.train.labels <- HTRU.train$confirmed
xgb.test.data <- model.matrix(~., data = HTRU.validation[,-c("confirmed")])
xgb.test.labels <- HTRU.validation$confirmed

xgb.train <- xgb.DMatrix(data = xgb.train.data, label = xgb.train.labels)
xgb.test <- xgb.DMatrix(data = xgb.test.data, label = xgb.test.labels)

# Fit the boosting model with the train dataset
fit <- xgb.train(data= xgb.train, booster = "gbtree", nrounds = 75, objective = "binary:logistic", eta=0.3
                 , gamma=0, max_depth=7, min_child_weight=1, subsample=1, colsample_bytree=1)

# Perform validation on the validation dataset
p_hat <- predict(fit, newdata = xgb.test, type = "response")

# Classify the candidate based on the cutoff
y_hat<- ifelse(p_hat > cutoff.xgb, 1, 0) %>% factor
confusionMatrix(y_hat,HTRU.validation$confirmed, positive = "1")