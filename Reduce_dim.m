%% Post Process the data to reduce dimensionality

%load('C:\MLSP\Seizure_Prediction\preprocessed\preprocessed_data.mat')
load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\preprocessed_data2.mat')
load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\labels.mat')


%% Remove features that are all the same
% stdevs = std(preprocessed_data');
% bad_feats = find(stdevs==0)';
% preprocessed_data(bad_feats,:) = [];

%% Split into training and testing
data = preprocessed_data';
rng('default');
cv = cvpartition(size(data,1), 'Holdout', 0.25);
idx = cv.test;
dataTrain = double(data(~idx, :));
dataTest = double(data(idx, :));
labels = labels';
labelsTrain = labels(~idx, :);
labelsTest = labels(idx, :);

fprintf('\nUsing All the Features\n\n');

%% Classification with all features

transformedTrain = dataTrain;
transformedTest = dataTest;

% SVM
MDL_svm = fitcsvm(transformedTrain, labelsTrain);
preds = predict(MDL_svm, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy SVM for all features : %2.3f\n', accuracy);

% kNN
MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
preds = predict(MDL_knn, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy kNN for all features : %2.3f\n', accuracy);

% Random Forest
MDL_rf = fitcensemble(transformedTrain, labelsTrain);
preds = predict(MDL_rf, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy random forest for all features : %2.3f\n', accuracy);


%% PCA

for k = [25, 50, 100]
    [coeff,transformedTrain,~] = pca(dataTrain, 'NumComponents', k);
    transformedTest = dataTest * coeff;
    % SVM
    MDL_svm = fitcsvm(transformedTrain, labelsTrain);
    preds = predict(MDL_svm, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy SVM for %d coefficients : %2.3f\n', k, accuracy);
    
    % kNN
    MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
    preds = predict(MDL_knn, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy kNN for %d coefficients : %2.3f\n', k, accuracy);
    
    % Random Forest
    MDL_rf = fitcensemble(transformedTrain, labelsTrain);
    preds = predict(MDL_rf, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy random forest for %d coefficients : %2.3f\n', k, accuracy);
end

%% Only DWT

load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\preprocessed_data2.mat')
load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\labels.mat')

preprocessed_data = preprocessed_data(1:576,:);

%% Split into training and testing
data = preprocessed_data';
rng('default');
cv = cvpartition(size(data,1), 'Holdout', 0.25);
idx = cv.test;
dataTrain = double(data(~idx, :));
dataTest = double(data(idx, :));
labels = labels';
labelsTrain = labels(~idx, :);
labelsTest = labels(idx, :);

fprintf('\nNow using only DWT Features\n\n');

%% Classification with all features

transformedTrain = dataTrain;
transformedTest = dataTest;

% SVM
MDL_svm = fitcsvm(transformedTrain, labelsTrain);
preds = predict(MDL_svm, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy SVM for all features : %2.3f\n', accuracy);

% kNN
MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
preds = predict(MDL_knn, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy kNN for all features : %2.3f\n', accuracy);

% Random Forest
MDL_rf = fitcensemble(transformedTrain, labelsTrain);
preds = predict(MDL_rf, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy random forest for all features : %2.3f\n', accuracy);


%% PCA

for k = [25, 50, 100]
    [coeff,transformedTrain,~] = pca(dataTrain, 'NumComponents', k);
    transformedTest = dataTest * coeff;
    % SVM
    MDL_svm = fitcsvm(transformedTrain, labelsTrain);
    preds = predict(MDL_svm, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy SVM for %d coefficients : %2.3f\n', k, accuracy);
    
    % kNN
    MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
    preds = predict(MDL_knn, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy kNN for %d coefficients : %2.3f\n', k, accuracy);
    
    % Random Forest
    MDL_rf = fitcensemble(transformedTrain, labelsTrain);
    preds = predict(MDL_rf, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy random forest for %d coefficients : %2.3f\n', k, accuracy);
end




%% Only EMD

load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\preprocessed_data2.mat')
load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\labels.mat')

preprocessed_data = preprocessed_data(577:576+656,:);

%% Split into training and testing
data = preprocessed_data';
rng('default');
cv = cvpartition(size(data,1), 'Holdout', 0.25);
idx = cv.test;
dataTrain = double(data(~idx, :));
dataTest = double(data(idx, :));
labels = labels';
labelsTrain = labels(~idx, :);
labelsTest = labels(idx, :);

fprintf('\n\nNow using only EMD Features\n\n');

%% Classification with all features

transformedTrain = dataTrain;
transformedTest = dataTest;

% SVM
MDL_svm = fitcsvm(transformedTrain, labelsTrain);
preds = predict(MDL_svm, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy SVM for all features : %2.3f\n', accuracy);

% kNN
MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
preds = predict(MDL_knn, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy kNN for all features : %2.3f\n', accuracy);

% Random Forest
MDL_rf = fitcensemble(transformedTrain, labelsTrain);
preds = predict(MDL_rf, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy random forest for all features : %2.3f\n', accuracy);


%% PCA

for k = [25, 50, 100]
    [coeff,transformedTrain,~] = pca(dataTrain, 'NumComponents', k);
    transformedTest = dataTest * coeff;
    % SVM
    MDL_svm = fitcsvm(transformedTrain, labelsTrain);
    preds = predict(MDL_svm, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy SVM for %d coefficients : %2.3f\n', k, accuracy);
    
    % kNN
    MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
    preds = predict(MDL_knn, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy kNN for %d coefficients : %2.3f\n', k, accuracy);
    
    % Random Forest
    MDL_rf = fitcensemble(transformedTrain, labelsTrain);
    preds = predict(MDL_rf, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy random forest for %d coefficients : %2.3f\n', k, accuracy);
end



%% Only EMD

load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\preprocessed_data2.mat')
load('C:\MLSP\Seizure_Prediction\preprocessed_30sec\labels.mat')

preprocessed_data = preprocessed_data(577+656:end,:);

%% Split into training and testing
data = preprocessed_data';
rng('default');
cv = cvpartition(size(data,1), 'Holdout', 0.25);
idx = cv.test;
dataTrain = double(data(~idx, :));
dataTest = double(data(idx, :));
labels = labels';
labelsTrain = labels(~idx, :);
labelsTest = labels(idx, :);

fprintf('\n\nNow using only WPD Features\n\n');

%% Classification with all features

transformedTrain = dataTrain;
transformedTest = dataTest;

% SVM
MDL_svm = fitcsvm(transformedTrain, labelsTrain);
preds = predict(MDL_svm, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy SVM for all features : %2.3f\n', accuracy);

% kNN
MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
preds = predict(MDL_knn, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy kNN for all features : %2.3f\n', accuracy);

% Random Forest
MDL_rf = fitcensemble(transformedTrain, labelsTrain);
preds = predict(MDL_rf, transformedTest);
accuracy = sum(preds == labelsTest) / length(labelsTest);
fprintf('Accuracy random forest for all features : %2.3f\n', accuracy);


%% PCA

for k = [25, 50, 100]
    [coeff,transformedTrain,~] = pca(dataTrain, 'NumComponents', k);
    transformedTest = dataTest * coeff;
    % SVM
    MDL_svm = fitcsvm(transformedTrain, labelsTrain);
    preds = predict(MDL_svm, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy SVM for %d coefficients : %2.3f\n', k, accuracy);
    
    % kNN
    MDL_knn = fitcknn(transformedTrain, labelsTrain, 'NumNeighbors', 5);
    preds = predict(MDL_knn, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy kNN for %d coefficients : %2.3f\n', k, accuracy);
    
    % Random Forest
    MDL_rf = fitcensemble(transformedTrain, labelsTrain);
    preds = predict(MDL_rf, transformedTest);
    accuracy = sum(preds == labelsTest) / length(labelsTest);
    fprintf('Accuracy random forest for %d coefficients : %2.3f\n', k, accuracy);
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% %% LDA
% 
% dataTrain = dataTrain';
% dataTest = dataTest';
% 
% Train_Data_full = cell(2,1);
% Train_Data_full{1} = dataTrain(:, [labelsTrain==0]');
% Train_Data_full{2} = dataTrain(:, [labelsTrain==1]');
% 
% w_l = zeros(size(dataTrain,1),2);
% n_l = zeros(1,2);
% full_temp = [];
% for class = 1:length(Train_Data_full)
%      M = Train_Data_full{class};
%      n_l(class) = size(M,2);
%      w_l(:,class) = mean(M,2);
%      full_temp = [full_temp, M]; 
% end
% w_bar = mean(full_temp,2);
% 
% % Compute Between Class Covariance
% clear full_temp
% Sb = zeros(length(w_bar),length(w_bar));
% for el = 1:length(n_l)
%     Sb = Sb + n_l(el)*(w_l(:,el)-w_bar)*((w_l(:,el)-w_bar)'); 
% end
% 
% 
% % Compute Cross Class Covariance
% Sw = zeros(length(w_bar),length(w_bar));
% for el = 1:length(n_l)
%     w = Train_Data_full{el};
%     for i = 1:n_l(el)
%         w_i_l = w(:,i);
%         Sw = Sw + (w_i_l - w_l(:,el))*((w_i_l - w_l(:,el))');
%     end
% end
% 
% % Compute eigenvalue decomposition for LDA
% %%
% Sw = double(Sw);
% [V,D] = eigs(Sb,Sw,length(n_l)-1);
% 
% % Project the I-vectors with the LDA matrix and normalize
% clear Sw Sb
% % m_l = matrix of means for each class's project I-vectors (23*24) where
% %       each column is the mean of that class
% m_l = zeros(length(n_l)-1,length(n_l));
% Train_projected_full = cell(size(Train_Data_full));
% for el = 1:length(Train_projected_full)
%     M = Train_Data_full{el};
%     proj = V'*M;                            % project
%     projnorm = proj./vecnorm(proj);         % normalize
%     Train_projected_full{el} = projnorm;
%     m_l(:,el) = mean(projnorm,2);           % compute mean of projections
%     m_l(:,el) = m_l(:,el)/norm(m_l(:,el));  % normalize the mean
% end
% 
% Test_Data_full = cell(2,1);
% Test_Data_full{1} = dataTrain(:, [labelsTest==0]');
% Test_Data_full{2} = dataTrain(:, [labelsTest==1]');
% 
% 
% % Test the classifier
% sum_correct = 0;
% sum_total_preds = 0;
% for el = 1:2
%     M = Test_Data_full{el};                  % look at all the examples of this class
%     proj = V'*M;                            % project
%     projnorm = proj./vecnorm(proj);         % normalize
%     scores = projnorm' * m_l;               % dot product between projection and each average class
%     [~, pred] = max(scores');               % vector of the prediction for each example
%     sum_correct = sum_correct + sum(pred==el);
%     sum_total_preds = sum_total_preds + length(pred);
% end
% 
% accuracy = sum_correct / sum_total_preds;
% fprintf('LDA Classifier Accuracy on Test Data : %2.3f%%\n\n',accuracy*100);
% 
% dataTrain = dataTrain';
% dataTest = dataTest';




