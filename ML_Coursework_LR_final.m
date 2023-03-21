clear all; clc; close all;

% Load Data
data=readtable('cleaned-bank-additional-csv_converted.xlsx');
% pulling in label encoded data
data_le=readtable('labelencoded-bank-additional-csv_converted.xlsx');

%% Data Partition
% split data into predictors - X and target variable - Y
cv = cvpartition(4119, 'KFold', 10);

X=data_le(:,[1:15]);
X=table2array(X);

Y=data_le(:,[16]);
Y=table2array(Y);
Y=Y+1;

%% Logistic Regression
% looping

Accuracy_array=[0 0 0 0 0 0 0 0 0 0];
Precision_array=[0 0 0 0 0 0 0 0 0 0];
Recall_array=[0 0 0 0 0 0 0 0 0 0];
F1_array=[0 0 0 0 0 0 0 0 0 0];

for i=1:cv.NumTestSets
    % split the train and test datasets using cvpartition
    X_train = X(cv.training(i), :);
    X_test = X(cv.test(i), :);
    Y_train = Y(cv.training(i), :);
    Y_test = Y(cv.test(i), :);

    % Train the model
    % B = mnrfit(X,Y) in which,
    % X should be an array type
    % Y should be an categorical type, if its a int/float/double type then it
    % should be in hierarchical B = mnrfit(X,Y,'Model','hierarchical')
    % in that case Y vector should have positive numbers, 0 may affect the
    % accuracy
    model_lr = mnrfit(X_train, Y_train, 'Model', "hierarchical");

    % Evaluate the model on the test data
    % pihat is probability of each category
    pihat = mnrval(model_lr, X_test);

    % yihat is probability of maximum of the 2 predict probabilities
    [prob,yihat] = max(pihat,[],2);

    % confusion matrix
    cm = confusionmat(Y_test, yihat);
    TP= cm(1,1);  
    FP = cm(2,1);  
    FN = cm(1,2);  
    TN = cm(2,2);
    
    Accuracy = (TP+TN)/(TP+TN+FP+FN); 
    Accuracy_array(i)=Accuracy;

    Precision = TP/(TP+FP);
    Precision_array(i)=Precision;

    Recall = TP/(TP+FN);
    Recall_array(i)=Recall;

    F1_Score = 2*(Recall * Precision) / (Recall + Precision);
    F1_array(i)=F1_Score;
end

%% Plot ROC Curvve
[fpr,tpr,thresh]= perfcurve(Y_test,pihat(:,1),1);
[fpr1,tpr1,thresh1]= perfcurve(Y_test,pihat(:,2),2);
figure(1);
plot(fpr,tpr,'linewidth',2);
hold on
plot(fpr1,tpr1,'linewidth',2);
hold off
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC')

%% Confusion Matrix Chart
cmc = confusionchart(Y_test,yihat);
cmc.RowSummary = 'row-normalized';
cmc.ColumnSummary = 'column-normalized';

%% Plot Metrics for CV
% Plot the actual and predicted values
plot(Accuracy_array, 'bo-') 
hold on 
plot(Precision_array, 'r*')
hold on
plot(Recall_array, 'g--')
hold on
plot(F1_array, 'c+')
legend('Accuracy', 'Precision','Recall','F1')
xlabel('Folds')
ylabel('efficiency Value')