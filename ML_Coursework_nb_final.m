clear all; clc; close all;

%Load Data
data=readtable('cleaned-bank-additional-csv_converted.xlsx');

%% Data Partition
% split data into predictors - X and target variable - Y
X=data(:,[1:15]);
Y=data(:,[16]);

% split data into trian and test data - using Cross Validation HoldOut
n= height(Y);
cv = cvpartition(n,'HoldOut', 0.2);

X_train=X(cv.training, :);
X_test=X(cv.test, :);
Y_train = Y(cv.training, :);
Y_test = Y(cv.test, :);

%% Naive Bayes regression
% Train the model
model = fitcnb(X_train,Y_train);

% predict Y
[Y_predict,scores]= predict(model,X_test);

% Measure Performance - Loss function and resubtitution loss
loss = loss(model, X_test, Y_test);
resubloss = resubLoss(model);

% confusion matrix values
cm = confusionmat(table2array(Y_test), Y_predict);
TP= cm(1,1);  
FP = cm(2,1);  
FN = cm(1,2);  
TN = cm(2,2);  
Accuracy = (TP+TN)/(TP+TN+FP+FN);  
Precision = TP/(TP+FP);  
Recall = TP/(TP+FN);  
F1_Score = 2*(Recall * Precision) / (Recall + Precision);

%% ROC curve

[Y_predict,scores]= predict(model,X_test);
size(scores);
rocObj = rocmetrics(table2array(Y_test),scores, model.ClassNames);
rocObj.AUC;
figure
plot(rocObj, ClassNames=model.ClassNames(1))

%% Confusion Matrix Chart
cmc = confusionchart(table2array(Y_test),Y_predict);
cmc.RowSummary = 'row-normalized';
cmc.ColumnSummary = 'column-normalized';

%%
%% Hyperparameter optimization

% Train the model
model_hp = fitcnb(X_train,Y_train,'OptimizeHyperparameters','auto');

% Predict Y with this model
Y_predict_hp = predict(model_hp,X_test);

% Measure Performance - resubtitution loss
resubloss_hp = resubLoss(model_hp);

% confusion matrix values
cm_hp = confusionmat(table2array(Y_test), Y_predict_hp);
TP_hp= cm_hp(1,1);  
FP_hp = cm_hp(2,1);  
FN_hp = cm_hp(1,2);  
TN_hp = cm_hp(2,2);  
Aaccuracy_hp = (TP_hp+TN_hp)/(TP_hp+TN_hp+FP_hp+FN_hp);  
Precision_hp = TP_hp/(TP_hp+FP_hp);  
Recall_hp = TP_hp/(TP_hp+FN_hp); 
F1_Score_hp = 2*(Recall_hp * Precision_hp) / (Recall_hp + Precision_hp); 

%% ROC curve with hyperperameter

[Y_predict_hp,scores_hp]= predict(model_hp,X_test);
size(scores);

rocObj = rocmetrics(table2array(Y_test),scores_hp, model.ClassNames);
rocObj.AUC;
figure
plot(rocObj, ClassNames=model.ClassNames(1))

%% Confusion Matrix Chart with hyperparameter
cmc = confusionchart(table2array(Y_test),Y_predict_hp);
cmc.RowSummary = 'row-normalized';
cmc.ColumnSummary = 'column-normalized';

%%