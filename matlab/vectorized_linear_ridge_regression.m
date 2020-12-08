clear all;

% Load the mat files
addpath('Data');
% A = age, T = task
% Therse are target variables and the file names are same.
A = load('age_240', '-mat');
T = load('task_240', '-mat');

% Input Data - The filename may change. 
% X = the input fMRI data (embedded or not)
input = load('conmat_240', '-mat');

% The mat files have data in the form of a table. So, use the coorect table
% names. 
% A and T are target variables representing the age and task variables. 
A = A.age;
T = T.mean_rxn;
% A and T have fixed table names. But X may differ. For full data the table
% name is full_conmat. It may vary for embedded data.
table = struct2cell(input);
X = cell2mat(table);

% The data has three dimensions,and there are 240 data points. 
% If the each data point has 268 rows and 268 columns, then it is full data.
% For full data only lower triangular matrix needs to be vectorized. 
% For other data, reshape is sufficient. Also the full data is in the shape
% 268*268*240. So, I hard coded which dimensions to look for.
if length(X(:,1,1)) == 268 && length(X(1,:,1)) == 268
    vec_X = vectorize_data(X);
else
    % There are 240 data pointa and the other dimensions may vary. So we
    % have to be careful when reshaping.
    a = length(X(:,1,1));
    b = length(X(1,:,1));
    c = length(X(1,1,:));
    
    if a == 240
        vec_X = reshape(X, 240, b*c);
    elseif b == 240
        vec_X = reshape(X, 240, a*c);
    else
        vec_X = reshape(X, 240, a*b);
    end
end

% If a bias term is needed for regression:
% vec_X = [ones(240,1) vec_X];

% Splitting the data into training and test data
X_train = vec_X(1:192,:);
X_test = vec_X(193:240,:);

% Y = [A T];
% Y_train = Y(1:192,:);
% Y_test = Y(193:240,:);
A_train = A(1:192,:);
A_test = A(193:240,:);
T_train = T(1:192,:);
T_test = T(193:240,:);

% The length of each vector
dim = length(X_train(1,:));


% Uncomment the following code and run to find the best regularization coefficients.
% Regularization coefficients on log scale
reg_coeffs = linspace(-3,3,62);


for i = 1:1:length(reg_coeffs)
    beta = 10^(reg_coeffs(i));
    
    % Closed form solutions for regression. For full data and some
    % embeddings, calculation of matrix inverse for closed form solution is
    % almost impossible. So, inbuilt functions are used. 
    % Uncomment the below two lines to use closed form solution. 
%     wc_age = ((X_train'*X_train + beta*eye(dim))^-1)*(X_train'*A_train);
%     wc_task = ((X_train'*X_train + beta*eye(dim))^-1)*(X_train'*T_train);

    wc_age = regress(A_train, X_train, beta);
    wc_task = regress(T_train, X_train, beta);
    
%     Y_tr_predicted = K_train*wc;
%     Y_test_predicted = K_test*wc;
    
    A_tr_predicted = X_train*wc_age;
    T_tr_predicted = X_train*wc_task;
    
    A_test_predicted = X_test*wc_age;
    T_test_predicted = X_test*wc_task;
    
    % MAE
    MAE_age_tr(i) = mean(abs(A_train - A_tr_predicted));
    MAE_age_test(i) = mean(abs(A_test - A_test_predicted));
    
    MAE_task_tr(i) = mean(abs(T_train - T_tr_predicted));
    MAE_task_test(i) = mean(abs(T_test - T_test_predicted));
    
    % Correlations:
    corr1 = corrcoef(A_train, A_tr_predicted);
    CORR_age_tr(i) = corr1(1,2);
    
    corr2 = corrcoef(A_test, A_test_predicted);
    CORR_age_test(i) = corr2(1,2);
    
    corr3 = corrcoef(T_train, T_tr_predicted);
    CORR_task_tr(i) = corr3(1,2);
    
    corr4 = corrcoef(T_test, T_test_predicted);
    CORR_task_test(i) = corr4(1,2);
end

% Plots to choose the best regularization coefficient
figure(1)
plot(reg_coeffs, CORR_age_test,'r', reg_coeffs, CORR_age_tr,'b')
title('Age Prediction - Correlation coefficients vs log(regularization coefficients)')
legend('test data', 'training data')
grid on

figure(2)
plot(reg_coeffs, MAE_age_test,'r', reg_coeffs, MAE_age_tr,'b')
title('Age prediction Mean Absoulte Error vs log(regularization coefficients)')
legend('test data', 'training data')
grid on

figure(3)
plot(reg_coeffs, CORR_task_test,'r', reg_coeffs, CORR_task_tr,'b')
title('Task Variable Prediction - Correlation coefficients vs log(regularization coefficients)')
legend('test data', 'training data')
grid on

figure(4)
plot(reg_coeffs, MAE_task_test,'r', reg_coeffs, MAE_task_tr,'b')
title('Task Variable Prediction - Mean Absolute Error vs log(regularization coefficients)')
legend('test data', 'training data')
grid on

% Finding the best regularization coefficient based on correlation between
% observed and predicted values:
[corr_age, i1] = max(CORR_age_test);
% i1 is the index of the maximum correlation for age prediction.
beta1 = reg_coeffs(i1);

[corr_task, i2] = max(CORR_task_test);
% i1 is the index of the maximum correlation for age prediction.
beta2 = reg_coeffs(i2);

% Regression, and finding mean absolute error and correlation coefficient
% wc_age = ((X_train'*X_train + beta*eye(192))^-1)*(X_train'*A_train);
% wc_task = ((X_train'*X_train + beta*eye(192))^-1)*(X_train'*T_train);
wc_age = regress(A_train, X_train, beta);
wc_task = regress(T_train, X_train, beta);
    
A_tr_predicted = X_train*wc_age;
T_tr_predicted = X_train*wc_task;
    
A_test_predicted = X_test*wc_age;
T_test_predicted = X_test*wc_task;
    
% MAE
% MAE for training and test dataset for predicting age.
MAE_age_tr = mean(abs(A_train - A_tr_predicted));
MAE_age_test = mean(abs(A_test - A_test_predicted));
    
% MAE for training and test dataset for predicting age.
MAE_task_tr = mean(abs(T_train - T_tr_predicted));
MAE_task_test = mean(abs(T_test - T_test_predicted));
    
% Correlation coefficients for age, task predictions on both training and test datasets:
corr1 = corrcoef(A_train, A_tr_predicted);
CORR_age_tr = corr1(1,2);
    
corr2 = corrcoef(A_test, A_test_predicted);
CORR_age_test = corr2(1,2);
    
corr3 = corrcoef(T_train, T_tr_predicted);
CORR_task_tr = corr3(1,2);
    
corr4 = corrcoef(T_test, T_test_predicted);
CORR_task_test = corr4(1,2);
