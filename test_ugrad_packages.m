%% Test ugrad package availability

X = rand(100,1);
XX = X*X';

% Signal Processing Toolbox
[imf,residual] = emd(X);

% Wavelet Toolbox
[cA,cD] = dwt(X,'db4');

% Statistics and Machine Learning Toolbox
coeff = pca(XX);

