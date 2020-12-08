clear all;
% Load the mat files
% A = age, T = task
% X = the input fMRI data (embedded or not)
X = load('conmat_240', '-mat');
A = load('age_240', '-mat');
T = load('task_240', '-mat');

% The mat files have data in the form of a table. So, use the coorect table
% names. 
A = A.age;
T = T.mean_rxn;
X = X.full_conmat;

% Vectorizing the data
for i = 1:1:240
   x = X(:,:,i);
   mask = tril(true(size(x)),-1);
   out = x(mask);
   
   vec_X(:,i) = out;
end

% Transpose to make the data into a set of row vectors
vec_X = vec_X';

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

% Generating the similarity Matrix Kernel.
for i = 1:1:length(X_train(:,1))
    xi = X_train(i,:);
    for j = 1:1:length(X_train(:,1))
        xj = X_train(j,:);
        sim = corrcoef(xi,xj);
        K_train(i,j) = sim(1,2);
    end
end

% Genertaing the test similarity matrix
for i = 1:1:length(X_test(:,1))
    xi = X_train(i,:);
    for j = 1:1:length(X_train(:,1))
        xj = X_train(j,:);
        sim = corrcoef(xi,xj);
        K_test(i,j) = sim(1,2);
    end
end

% The method is kernel linear ridge regression. 
% The method to find the best L2 regularization  coefficient.

% Uncomment the following code and run to find the best regularization
% coefficients.
% Regularization coefficients on log scale
% reg_coeffs = linspace(-3,3,62);
% 
% for i = 1:1:length(reg_coeffs)
%     beta = 10^(reg_coeffs(i));
%     
%     % Closed form solutions for regression (we have 192*192 matrix. So I
%     % used closed form solutions.
%     wc_age = ((K_train'*K_train + beta*eye(192))^-1)*(K_train'*A_train);
%     wc_task = ((K_train'*K_train + beta*eye(192))^-1)*(K_train'*T_train);
%     
% %     Y_tr_predicted = K_train*wc;
% %     Y_test_predicted = K_test*wc;
%     
%     A_tr_predicted = K_train*wc_age;
%     T_tr_predicted = K_train*wc_task;
%     
%     A_test_predicted = K_test*wc_age;
%     T_test_predicted = K_test*wc_task;
%     
%     % MAE
%     MAE_age_tr(i) = mean(abs(A_train - A_tr_predicted));
%     MAE_age_test(i) = mean(abs(A_test - A_test_predicted));
%     
%     MAE_task_tr(i) = mean(abs(T_train - T_tr_predicted));
%     MAE_task_test(i) = mean(abs(T_test - T_test_predicted));
%     
%     % Correlations:
%     corr1 = corrcoef(A_train, A_tr_predicted);
%     CORR_age_tr(i) = corr1(1,2);
%     
%     corr2 = corrcoef(A_test, A_test_predicted);
%     CORR_age_test(i) = corr2(1,2);
%     
%     corr3 = corrcoef(T_train, T_tr_predicted);
%     CORR_task_tr(i) = corr3(1,2);
%     
%     corr4 = corrcoef(T_test, T_test_predicted);
%     CORR_task_test(i) = corr4(1,2);
% end
% 
% % Plots to choose the best regularization coefficient
% figure(1)
% plot(reg_coeffs, CORR_age_test,'r', reg_coeffs, CORR_age_tr,'b')
% title('Correlation coeffs Age')
% grid on
% 
% figure(2)
% plot(reg_coeffs, MAE_age_test,'r', reg_coeffs, MAE_age_tr,'b')
% title('Correlation coeffs Age')
% grid on
% 
% figure(3)
% plot(reg_coeffs, CORR_task_test,'r', reg_coeffs, CORR_task_tr,'b')
% grid on
% 
% figure(4)
% plot(reg_coeffs, MAE_task_test,'r', reg_coeffs, MAE_task_tr,'b')
% grid on

% Regression, and finding mean absolute error and correlation coefficient
wc_age = ((K_train'*K_train + beta*eye(192))^-1)*(K_train'*A_train);
wc_task = ((K_train'*K_train + beta*eye(192))^-1)*(K_train'*T_train);
    
A_tr_predicted = K_train*wc_age;
T_tr_predicted = K_train*wc_task;
    
A_test_predicted = K_test*wc_age;
T_test_predicted = K_test*wc_task;
    
% MAE
% MAE for training and test dataset for predicting age.
MAE_age_tr = mean(abs(A_train - A_tr_predicted));
MAE_age_test = mean(abs(A_test - A_test_predicted));
    
% MAE for training and test dataset for predicting age.
MAE_task_tr(i) = mean(abs(T_train - T_tr_predicted));
MAE_task_test(i) = mean(abs(T_test - T_test_predicted));
    
% Correlation coefficients for age, task predictions on both training and test datasets:
corr1 = corrcoef(A_train, A_tr_predicted);
CORR_age_tr = corr1(1,2);
    
corr2 = corrcoef(A_test, A_test_predicted);
CORR_age_test = corr2(1,2);
    
corr3 = corrcoef(T_train, T_tr_predicted);
CORR_task_tr = corr3(1,2);
    
corr4 = corrcoef(T_test, T_test_predicted);
CORR_task_test = corr4(1,2);