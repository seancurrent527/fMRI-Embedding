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

% Reshaping the dataset to match the input layer preferences in MATLAB. 
% For deep learning toolbox in MATLAB, the inputs are either images or
% sequences. There was a featureInputLayer previously (2017), but that has
% been removed recently.
% c = number of columns in each data point.
c = length(vec_X(1,:));
vec_X2 = reshape(vec_X, 1, c, 1, 240);

% Splitting into train and test datasets (80% and 20%)
X_train = vec_X2(:,:,:,1:192);
X_test = vec_X2(:,:,:,193:240);


% Observed test data:
Age_observed_train = A(1:192,:);
Task_observed_train = T(1:192,:);
Age_observed_test = A(193:240,:);
Task_observed_test = T(193:240,:);

% Initializing the Layers for the FCNN - regression layer at the end
% separate networks are used to predict age and task.
% This is the set of layers for predicting age
layers = [
    imageInputLayer([1 35778 1])
    fullyConnectedLayer(10)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(8)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

layers2 = [
    imageInputLayer([1 35778 1])
    fullyConnectedLayer(10)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(8)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];


miniBatchSize  = 18;
validationFrequency = floor(192/miniBatchSize);

% Training options for the FCNN 
% options1 = trainingOptions('adam', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',40, ...
%     'InitialLearnRate',5e-2, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',20, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',{X_test,Age_observed_test}, ...
%     'ValidationFrequency',validationFrequency, ...
%     'Plots','training-progress', ...
%     'Verbose',false);

options1 = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',5e-2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{X_test,Age_observed_test}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

% options2 = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',100, ...
%     'InitialLearnRate',5e-2, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',20, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',{X_test,Task_observed_test}, ...
%     'ValidationFrequency',validationFrequency, ...
%     'Plots','training-progress', ...
%     'Verbose',false);

options2 = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

% Network 1 -  for age:
net1 = trainNetwork(X_train,Age_observed_train,layers,options1);
% Network 2 -  for task variable:
net2 = trainNetwork(X_train,Task_observed_train,layers2,options2);

% Network performance measures based on test data:
Age_predicted_test = predict(net1,X_test);
Age_prediction_error = Age_observed_test - Age_predicted_test;

Task_predicted_test = predict(net2,X_test);
Task_prediction_error = Task_observed_test - Task_predicted_test;

% Mean Absolute Error:
MAE_age_test = mean(abs(Age_prediction_error));
MAE_task_test = mean(abs(Task_prediction_error));

% Correlation:
corr1 = corrcoef(Age_predicted_test, Age_observed_test);
corr_age_test = corr1(1,2);

corr2 = corrcoef(Task_predicted_test, Task_observed_test);
corr_task_test = corr2(1,2);

% The above 2 correlation values and the 2 mae values are used to measure
% the performance.


