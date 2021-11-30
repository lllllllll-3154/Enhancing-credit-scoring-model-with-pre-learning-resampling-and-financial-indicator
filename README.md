# Enhancing credit scoring model with pre-learning resampling and financial indicator
# FITE7410-A GROUP11
## Author: Chen Qixun, Luo Xingjian, Luan Jianduo, Pu Zihao


## Background 
We retrieve the data set from Kaggle https://www.kaggle.com/mishra5001/credit-card?select=application_data.csv. The data was collected from one social experiment providing public inferences of how a person applying for loan can get it completed in a minimal amount of time. Our main object is to detect the fraud case.

## objectives
1. To build a balanced dataset for training with fewer noises.
2. Keep the interpretability of our model.
3. Find one suitable method for feature selection.
4. Find other evalution for scoring the model.

## Procedures
[1.EDA](#eda)

[2.Data Cleaning](#dc)

[3.Resampling and cv](#re)

[4.Modeling](#mo)



### *required package
library(ggplot2)

library(dplyr)

library(corrplot)

library(nnet)

library(caret)

library("smotefamily")

library(randomForest)

library(pROC)

library(klaR)

##  <h3 id='eda'>1.EDA</h3>
  As usual structured data, this data set contains numeric features and character features. Some of character features are hierarchical, so we need to encode them into a properly numeric way to represent the hierarchy. 
Besides, we notice that there are a great amount of ‘NA’ values in some columns. We would like to complete those values or get rid of them, which depends on the specific situation.
Here we show three typical distributions of features. One is the skewed distribution.
The second one is the approximately bell-shaped distribution. The last one is the imbalanced distribution. 

![image](https://user-images.githubusercontent.com/63034312/144006753-a5198f3c-3bcf-492a-bdde-71cff61a7ee3.png)
![image](https://user-images.githubusercontent.com/63034312/144006771-5c795b23-b30b-4cb4-99da-09d23a3f0388.png)
![image](https://user-images.githubusercontent.com/63034312/144006810-8fa4b1ac-8ec4-4a3a-a739-43e3cbc46197.png)
![image](https://user-images.githubusercontent.com/63034312/144006819-5b1bcfd3-f8c9-4c5b-809f-2a9d4a1846fd.png)
![image](https://user-images.githubusercontent.com/63034312/144006826-be142bd0-a1c5-46b4-abd0-1036d49e5c08.png)
![image](https://user-images.githubusercontent.com/63034312/144006832-1494465b-914f-4fa5-9167-a3f92a487f35.png)

Correlation

![image](https://user-images.githubusercontent.com/63034312/144007003-148286aa-9a58-4553-804e-91b3fbf2d041.png)
```
bina_idx <- function(dt){
  indices <- c()
  for (i in colnames(dt)){
    if(length(unique(dt[i])) <= 2){
      idices = c(indices, i)
    }
  }
  indices
}
# columns of binary features
bina_indices <- bina_idx(dt.raw)
# scaled data
dt.scaled <- dt.raw %>% select(-bina_indices) %>% mutate_if(is.numeric, scale)
dt.scaled <- data.frame(dt.scaled, select(dt.raw, bina_indices)) %>% mutate_if(is.character, as.factor)
# numeric features
dt.num <- dt.scaled %>% select(is.numeric)
plot_all <- function(dt, func){
  nums <- length(colnames(dt))
  par(ceiling(nums/4), 4)
  for (i in 1:nums) {
    func(dt[, i], main=colnames(dt)[i])
  }
}
plot_all(dt.num, boxplot)
plot_all(dt.num, hist)
dt.nona <- dt.num[complete.cases(dt.num), ]
png('./corrplot1.png', width=1920, height=1080)
cor_mat <- cor(dt.nona)
corrplot(cor_mat)
dev.off()
```


  
##  <h3 id='dc'>2.Data Cleaning</h3>
+ Firstly, the original data has lots of missing values. Therefore, any observation having a missing value is deleted. 
+ Then, after checking the values of each attribute, we split the data into five parts, namely useless variables, non-ordered variables, ordered variables, zero-one variables    and numeric variables. 
+ For useless variables, they are mainly of character type and almost each observation has different values. We deal with them by deleting them.

```
na_mat = is.na(data)
n_row_na = matrix(0,1,1)
for (i in 1:nrow(data)) {
  if(sum(na_mat[i,])>0){
    n_row_na = cbind(n_row_na,i)
  }
}
n_row_na
data_new = data[-n_row_na,]
table(data_new$TARGET)
save(data_new,file = "D:\\HK文件\\my DS\\7401 Finance Fraud\\project\\data_new.RData")
data = fread('/Users/luanjianduo/Documents/FITE7410 financial fraud/archive/application_data.csv')
str(data)
nr = nrow(data)
data = load('/Users/luanjianduo/Desktop/data_new.RData')
dim(data_new)
head(data_new[1:6])
rownames(data_new) = 1:nrow(data_new)
# split into 5 parts
data_drop = subset(data_new, select = c(1, 5, 23, 26, 97, 99, 102, 105, 107, 111))
dim(data_drop)
data_nonorder = subset(data_new, select = c(4, 12, 13, 15, 16, 29, 33, 41, 87, 88, 90))
dim(data_nonorder)
data_order = subset(data_new, select = c(7, 14, 31, 32))
dim(data_order)
data_zerone = subset(data_new, select = c(3, 6, 24, 25, 27, 28, 35:40, 91, 98, 100, 101,
                                          103, 104, 106, 108:110, 112:116))
dim(data_zerone)
data_num = subset(data_new, select = -c(1, 5, 23, 26, 97, 99, 102, 105, 107, 111,
                                        4, 12, 13, 15, 16, 29, 33, 41, 7, 14, 31, 32, 3, 6, 24, 25, 27, 28, 35:40, 87, 88, 90, 91, 98, 100, 101,
                                        103, 104, 106, 108:110, 112:116))
dim(data_num)
# for nonorder
library(nnet)
str(data_nonorder)
for (i in 1:ncol(data_nonorder)){
  data_nonorder[,i] = as.factor(data_nonorder[,i])
}
head(data_nonorder[1:6])
ls = list()
for (i in 1:ncol(data_nonorder)){
  ls[[i]] = class.ind((data_nonorder[,i]))
}
for (i in 1:ncol(data_nonorder)){
  for (j in 1:nrow(ls[[i]])){
    if (as.numeric(ls[[i]][j,ncol(ls[[i]])]) == 1){
      ls[[i]][j,1:(ncol(ls[[i]])-1)] = 1
    }
  }
}
data_newnonorder = data.frame(row.names = 1:nrow(data_nonorder))
for (i in 1:ncol(data_nonorder)){
  data_newnonorder = cbind(data_newnonorder,subset(as.data.frame(ls[[i]]), select = -c(ncol(ls[[i]]))))
}
head(data_newnonorder)
dim(data_newnonorder)
# for order
head(data_order)
table(data_order$REGION_RATING_CLIENT_W_CITY)
data_order$CNT_CHILDREN = as.numeric(factor(data_order$CNT_CHILDREN, 
                                            levels = c('0','1','2','3','4','5')))
data_order$NAME_EDUCATION_TYPE = as.numeric(factor(data_order$NAME_EDUCATION_TYPE, 
                                                   levels = c('Lower secondary','Secondary / secondary special',
                                                              'Incomplete higher','Higher education','Academic degree')))
data_order$REGION_RATING_CLIENT = as.numeric(factor(data_order$REGION_RATING_CLIENT, 
                                                    levels = c('1','2','3')))
data_order$REGION_RATING_CLIENT_W_CITY = as.numeric(factor(data_order$REGION_RATING_CLIENT_W_CITY, 
                                                           levels = c('1','2','3')))
dim(data_order)
# for zerone
colnames(data_zerone)
for (i in 1:ncol(data_zerone)){
  data_zerone[,i] = as.numeric(factor(data_zerone[,i]))-1
}
dim(data_zerone)
head(data_zerone,10)
# combine
data_clean = cbind(data_newnonorder, data_order, data_zerone, data_num)
dim(data_clean)
save(data_clean, file = '/Users/luanjianduo/Desktop/data_clean.RData')
head(data_clean[1:6])
str(data_clean)
data_temp = data_clean
head(data_temp[131])
for (i in c(1:115,106:131)){
  data_temp[,i] = factor(data_temp[,i])
}
data_temp$TARGET = factor(data_temp$TARGET)
str(data_temp$TARGET)
save(data_temp, file = '/Users/luanjianduo/Desktop/data_clean.RData')
data_temp$FLAG_OWN_REALTY
data_temp$FLAG_OWN_REALTY
#
table(data_temp$FONDKAPREMONT_MODE)  # nonorder
table(data_temp$HOUSETYPE_MODE)  # nonorder
table(data_temp$WALLSMATERIAL_MODE) # nonorder
table(data_temp$EMERGENCYSTATE_MODE) # zerone
```

##  <h3 id='re'>3.Resampling and cv</h3>
We firstly divided the clean data into train set and test set. And then balanced train data set with three popular methods (SMOTE, ADASYN, BORDERLINE SMOTE). Then do a cross validation on the resampled train dataset. The modeling will be conducted on the train set and evaluation will be on the test set.

+ codes for resasmpling and cv:
```
##unlist function
unli <- function(x){
  a <- numeric()
  for(i in 1:4){
    a <- cbind(a,x[i])
  }
  a <- unlist(a)
  return(a)
}

#create train set and test set
index <- createFolds(1:nrow(data_clean), k = 4, list = TRUE, returnTrain = FALSE)
index_train_list <- index[-1]
index_test_list <- index[1]
index_train <- unli(index_train_list)
index_test <- unli(index_test_list)
dataset_train <- data_clean[index_train,]
dataset_test <- data_clean[index_test,]
subset_test <- subset(dataset_test,select=-c(TARGET))
dataset_test <- cbind(subset_test,dataset_test$TARGET)
# the test set for final evaluation
save(dataset_test,file = "dataset_test")
data_clean <- dataset_train


##create index for 5 folds cv
index <- createFolds(1:nrow(data_clean), k = 5, list = TRUE, returnTrain = FALSE)
for(i in 1:length(index)){
  
  #select 4 folds for training dataset
  
  #get index for train set and test set
  index_train_list <- index[-i]
  index_test_list <- index[i]
  
  #unlist the index 
  index_train <- unli(index_train_list)
  index_test <- unli(index_test_list)
  
  #form the train set data in a cv
  dataset_train <- data_clean[index_train,]
  dataset_test <- data_clean[index_test,]
  
  #move the target class to the last column
  subset_test <- subset(dataset_test,select=-c(TARGET))
  dataset_test <- cbind(subset_test,dataset_test$TARGET)
  
  #smote
  library("smotefamily")
  #resampling with smote
  dataset_train_smote <- SMOTE(subset(dataset_train,select=-c(TARGET)),target = dataset_train[,"TARGET"])
  #define and save train RData name
  dataset_train_smote_file <- paste("dataset_train_smote",as.character(i),".RData",sep='')
  #get the train data
  dataset_train_smote <- dataset_train_smote$data
  #combine the 4-folds train set and the test set
  dataset_train_smote <- list(dataset_train_smote,dataset_test)
  #save the file 
  save(dataset_train_smote,file = dataset_train_smote_file)
  
  #adas
  #resampling with ada
  dataset_train_ada <- ADAS(subset(dataset_train,select=-c(TARGET)),target = dataset_train[,"TARGET"])
  dataset_train_ada_file <- paste("dataset_train_ada",as.character(i),".RData",sep='')
  dataset_train_ada <- dataset_train_ada$data
  dataset_train_ada <- list(dataset_train_ada,dataset_test)
  save(dataset_train_ada,file = dataset_train_ada_file)
  
  #borderline
  #resampling with boderlinesmote
  dataset_train_blsmote <- BLSMOTE(subset(dataset_train,select=-c(TARGET)),target = dataset_train[,"TARGET"])
  dataset_train_blsmote_file <- paste("dataset_train_blsmote",as.character(i),".RData",sep='')
  dataset_train_blsmote <- dataset_train_blsmote$data
  dataset_train_blsmote <- list(dataset_train_blsmote,dataset_test)
  save(dataset_train_blsmote,file = dataset_train_blsmote_file)
  

}


```

##  <h3 id='mo'>4.Modeling</h3>
+ Pre-Learning Resampling
The first step of this research was using the assistant classifiers to perform pre-learning resampling on the above three oversampling train sets, whose purpose was to reduce the original noises in dataset and the generated noises caused by the oversampling methods. Although improved methods such as Borderline-SMOTE and ADA-SYN can moderate the overlap phenomenon to some extent, overlap problem can not be eliminated absolutely. 
  - Naive Bayes Distance: The Naive Bayes classifier gives the predicted probability of taking different values for the test feature Y^' in the pre-learning resampling step. 
  - Attention and resample method: Regarding the Naive Bayes discussed before and any train set point (X^',Y^'), the larger its Distance_bayes was, the greater the probability of the model's misjudgment for the sample was.

```
#output mat
Rf_outcome = matrix(0,4,1)


#####pre-learning resample function#####
left_sample = function(noisy_score){
  N = floor(length(noisy_score)/3)
  noisy_rank = rank(noisy_score)
  rank1 = which(noisy_rank>N*2)
  sample1 = train_set[rank1,]
  sample1 = rbind(sample1,sample1,sample1)
  rank2 = which(noisy_rank<=(N*2)&noisy_rank>N)
  sample2 = train_set[rank2,]
  sample2 = rbind(sample2,sample2)
  rank3 = which(noisy_rank<=N)
  sample3 = train_set[rank3,]
  data_woe_noise = rbind(sample1,sample2,sample3)
  return(data_woe_noise)
  
  ####class type adjust#####
  train_set = as.data.frame(dataset_train[1])
  test_set = as.data.frame(dataset_train[2])
  train_set$class = as.factor(train_set$class)
  test_set$dataset_test.TARGET = as.factor(test_set$dataset_test.TARGET)

  #####v_name adjust and print#####
  for (i in 1:(ncol(train_set)-1)) {
    names(train_set)[i] = paste("V",as.character(i),sep = "_")
  }
  for (i in 1:(ncol(test_set)-1)) {
    names(test_set)[i] = paste("V",as.character(i),sep = "_")
  }
  v_name = "aa"
  for (i in 1:ncol(train_set)) {
    v_name = paste(v_name,names(train_set)[i],sep = "+")
  }
  
  #####Naive Bayes#######
  train_set_bayes = train_set[,-c(142,145,146)]
  test_set_bayes = test_set[,-c(142,145,146)]
  Bayes = NaiveBayes(class~.,data = train_set_bayes)
  
  #### Pre-learning resampling####
  bayes_train_predict = predict(Bayes,newdata = train_set_bayes[,-ncol((test_set_bayes))])
  bayes_train_outcome = cbind(train_set_bayes$class,bayes_train_predict$posterior[,2])
  index_na_train = which(is.na(bayes_train_outcome[,2])==TRUE)
  if(length(index_na_train)!=0){
    train_set_bayes = train_set_bayes[-index_na_train,]
    bayes_train_outcome = bayes_train_outcome[-index_na_train,]
    bayes_train_predict$class = bayes_train_predict$class[-index_na_train]
  }
  
  
  train_pre = as.numeric(bayes_train_predict$class)-1
  weight = matrix(0,length(train_pre),1)
  for (i in 1:length(train_pre)) {
    if(train_pre[i]==0){
      weight[i,] = bayes_train_predict$posterior[i,2]
    }else{
      weight[i,] = bayes_train_predict$posterior[i,1]
    }
  }
  bayes = left_sample(weight)  
  train_resample = bayes
  save(train_resample,test,v_sort,file=paste(output.dir,'\\',file.title,'_bayes.RData',sep=''))
  rm(train_resample)
  
}

```

+ Variables selection with MDA
In the variable-selection step, the MDA given by Random Forest algorithm was used to sort features, and then the features were reduced. 

```
#####random forest####
  rf_outcome = matrix(0,2,4)
  rf_outcome[1,1] = file.title
  rf_outcome[1,4] = "AUC"
  
  sjsl = randomForest(class ~ V_1+V_2+V_3+V_4+V_5+V_6+V_7+V_8+V_9+V_10+V_11+V_12+V_13+V_14+V_15+V_16+V_17+V_18+V_19+V_20+V_21+V_22+V_23+V_24+V_25+V_26+V_27+V_28+V_29+V_30+V_31+V_32+V_33+V_34+V_35+V_36+V_37+V_38+V_39+V_40+V_41+V_42+V_43+V_44+V_45+V_46+V_47+V_48+V_49+V_50+V_51+V_52+V_53+V_54+V_55+V_56+V_57+V_58+V_59+V_60+V_61+V_62+V_63+V_64+V_65+V_66+V_67+V_68+V_69+V_70+V_71+V_72+V_73+V_74+V_75+V_76+V_77+V_78+V_79+V_80+V_81+V_82+V_83+V_84+V_85+V_86+V_87+V_88+V_89+V_90+V_91+V_92+V_93+V_94+V_95+V_96+V_97+V_98+V_99+V_100+V_101+V_102+V_103+V_104+V_105+V_106+V_107+V_108+V_109+V_110+V_111+V_112+V_113+V_114+V_115+V_116+V_117+V_118+V_119+V_120+V_121+V_122+V_123+V_124+V_125+V_126+V_127+V_128+V_129+V_130+V_131+V_132+V_133+V_134+V_135+V_136+V_137+V_138+V_139+V_140+V_141+V_142+V_143+V_144+V_145+V_146+V_147+V_148+V_149+V_150+V_151+V_152+V_153+V_154+V_155+V_156+V_157+V_158+V_159+V_160+V_161+V_162+V_163+V_164+V_165+V_166+V_167+V_168+V_169+V_170+V_171+V_172+V_173+V_174+V_175+V_176+V_177+V_178+V_179+V_180+V_181+V_182+V_183+V_184+V_185+V_186+V_187+V_188+V_189+V_190+V_191+V_192+V_193+V_194+V_195+V_196+V_197+V_198+V_199+V_200+V_201+V_202+V_203+V_204+V_205+V_206+V_207+V_208+V_209+V_210+V_211+V_212+V_213+V_214+V_215, data = train_set, importance = TRUE,ntree =100)
  
  #########variable selection####
  if(p == 1){
    MDA = sjsl$importance[,3]
    index_v_MDA = which(MDA>0.001)
    MDA = as.data.frame(sort(MDA))
    varImpPlot(sjsl, main = "variable importance")
  
     #change test & train

  }
```


+ logistic regression
The fourth step of this research was, based on the consideration of statistics and finance, using logistic regression method to predict whether customers will be default. In the actual loan business of companies. From the perspective of customers, after the loan application rejected, they often hope to get a reason for rejection. Coefficients of logistic regression can be used to clarify rejection reasons. By this way, companies can better communicate with the customer. Therefore, logistics regression was used as chief classifier. 

```
 ## we use fixed selected variable over 5-fold
    test_set = test_set[,index_v_MDA]
    train_set = train_set[,index_v_MDA]
  
  logistics = glm(class ~ V_1+V_2+V_3+V_4+V_5+V_6+V_7+V_8+V_9+V_10+V_11+V_12+V_13+V_14+V_15+V_16+V_17+V_18+V_19+V_20+V_21+V_22+V_23+V_24+V_25+V_26+V_27+V_28+V_29+V_30+V_31+V_32+V_33+V_34+V_35+V_36+V_37+V_38+V_39+V_40+V_41+V_42+V_43+V_44+V_45+V_46+V_47+V_48+V_49+V_50+V_51+V_52+V_53+V_54+V_55+V_56+V_57+V_58+V_59+V_60+V_61+V_62+V_63+V_64+V_65+V_66+V_67+V_68+V_69+V_70+V_71+V_72+V_73+V_74+V_75+V_76+V_77+V_78+V_79+V_80+V_81+V_82+V_83+V_84+V_85+V_86+V_87+V_88+V_89+V_90+V_91+V_92+V_93+V_94+V_95+V_96+V_97+V_98+V_99+V_100+V_101+V_102+V_103+V_104+V_105+V_106+V_107+V_108+V_109+V_110+V_111+V_112+V_113+V_114+V_115+V_116+V_117+V_118+V_119+V_120+V_121+V_122+V_123+V_124+V_125+V_126+V_127+V_128+V_129+V_130+V_131+V_132+V_133+V_134+V_135+V_136+V_137+V_138+V_139+V_140+V_141+V_142+V_143+V_144+V_145+V_146+V_147+V_148+V_149+V_150+V_151+V_152+V_153+V_154+V_155+V_156+V_157+V_158+V_159+V_160+V_161+V_162+V_163+V_164+V_165+V_166+V_167+V_168+V_169+V_170+V_171+V_172+V_173+V_174+V_175+V_176+V_177+V_178+V_179+V_180+V_181+V_182+V_183+V_184+V_185+V_186+V_187+V_188+V_189+V_190+V_191+V_192+V_193+V_194+V_195+V_196+V_197+V_198+V_199+V_200+V_201+V_202+V_203+V_204+V_205+V_206+V_207+V_208+V_209+V_210+V_211+V_212+V_213+V_214+V_215,data = train_set,family = binomial('logit'))
  
  coef_i = as.data.frame(logistics$coefficients)
  coef = cbind(coef,coef_i)
```

# Evaluation and outcomes

we use two methods to evaluate the score of the model:

+ Statistical Indicators： ROC curve and AUC score
+ Actual financial loss and its rate: And maximum cost/actual cost.

```
#####financial loss####
#not pay back
index_default = which(data$TARGET==1)
cost_default = mean(data$AMT_CREDIT[index_default])
#profit get
index_true = which(data$TARGET==0)
cost_profit = mean(data$AMT_CREDIT[index_true])*0.1

file = list.files(input.dir)
total.start = Sys.time()
coef = data.frame()
for (p in seq(1,length(file)))  {
  temp.start <- Sys.time()
  load(paste(input.dir,'\\',file[p],sep=''))
  file.title = substring(file[p],1,nchar(file[p])-6)
  coef = apply(coef,2,sum)
  coef = coef/5
  test_set = dataset_test[,index_v_MDA]
  pred = dataset_test%*%t(coef)
  logit = function(x){
    1/(1+exp(-x))
  }
  pred_logit_class = apply(pred, 1, logit)

  predtrue_logit= data.frame(pred_y=pred_logit_class,true_y=test_set$dataset_test.TARGET)
table_logit= table(test_set$dataset_test.TARGET,pred_logit_class,dnn=c("Actual value","Prediction"))

#ROC
logit_roc = roc(test_set$dataset_test.TARGET,as.numeric(pred_logit))
plot(logit_roc, print.auc=TRUE, auc.polygon=TRUE, legacy.axes = TRUE,grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='logit_ROC')

}

```

