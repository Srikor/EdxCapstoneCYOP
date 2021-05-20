# Download the required packages from CRAN and install
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(ROCR)) install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(boot)) install.packages("boot", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

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
HTRU.test <- HTRU[test.index,]

# Fit a logistic regression model first to verify its performance. The prediction result of logistic regression
# is the probability of a candidate being a pulsar.
glm.fit <- glm(confirmed ~ ., data = HTRU.train, family = "binomial")
p_hat.glm <- predict(glm.fit, newdata = HTRU.test, type = "response")

# With a default cutoff of 0.5 make prediction on test set and create the confusion matrix.
y_hat.glm <- ifelse(p_hat.glm > 0.5, 1, 0) %>% factor
confusionMatrix(y_hat.glm,HTRU.test$confirmed, positive = "1")

# Verify if cross validation can impprove the model performance
# Perform cross-validation to find out the ideal cutoff based on the highest f-score. 
cutoff_seq.glm <- seq(0, 1, by = 0.05)
p_hat.glm.df <- data.frame()
for(p in cutoff_seq.glm){
  y_hat.glm <- ifelse(p_hat.glm > p, 1, 0) %>% factor
  conf_mat <- confusionMatrix(y_hat.glm,HTRU.test$confirmed, positive = "1")
  df_row <- data.frame(cutoff = p, Sens = conf_mat$byClass["Sensitivity"], Spec = conf_mat$byClass["Specificity"], 
                       fscore = 2* ((conf_mat$byClass["Sensitivity"]*conf_mat$byClass["Specificity"])/
                                      (conf_mat$byClass["Sensitivity"]+conf_mat$byClass["Specificity"])))
  p_hat.glm.df <- rbind(p_hat.glm.df,df_row)
}
cutoff.glm <- p_hat.glm.df[which(p_hat.glm.df$fscore == max(p_hat.glm.df$fscore)),]$cutoff

# Use the cutoff value derived from cross-validation to make prediction on the test set once again
y_hat.glm <- ifelse(p_hat.glm > cutoff.glm, 1, 0) %>% factor
conf_mat <- confusionMatrix(y_hat.glm,HTRU.test$confirmed, positive = "1")

# Get the important statistics of the testing to eventually compare and finalize the model.
pred <- prediction(as.numeric(p_hat.glm), HTRU.test$confirmed)
auc.glm <- as.numeric(performance(pred,  measure  = "auc")@y.values)
model.results <- data.frame(Model = "glm", CutOff = cutoff.glm, sensitivity = conf_mat$byClass["Sensitivity"], 
                            Acc = conf_mat$overall["Accuracy"], PPV = conf_mat$byClass["Pos Pred Value"],
                            AUC = auc.glm)

model.results %>% knitr::kable()







