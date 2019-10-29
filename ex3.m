% N defines the sub-window size
N = 32;
% Form mean color feature of training data
F = cifar_10_features(tr_data(1:50000,:), N);
% Form mean color feature of test data
f = cifar_10_features(te_data(1:10000,:), N);

%% Bayesian classifier. NAIVE CLASSIFICATION

% Compute the normal distribution parameters
[mu, sigma, p] = cifar_10_bayes_learn(F(1:50000,:), tr_labels(1:50000,:));
% Returns the Bayesian optimal class
c = cifar_10_bayes_classify(f(1:10000,:), mu, sigma, p);
% Evaluate the prediction
cifar_10_evaluate(c, te_labels(1:10000));

%% Bayesian classifier with better pdf. MULTIVARIATE CLASSIFICATION

% Compute the normal distribution parameters
[MU, SIGMA, p] = cifar_10_bayes_learn_better(F(1:50000,:), tr_labels(1:50000,:));
% Returns the Bayesian optimal class
c = cifar_10_bayes_classify_better(f(1:10000,:), MU, SIGMA, p);
% Evaluate the prediction
cifar_10_evaluate(c, te_labels(1:10000));

%% Bayesian with extended features. SUB-IMAGES

% N defines the sub-window size
N = 16;
% Form mean color feature of training data
F = cifar_10_features(tr_data(1:50000,:), N);
% Form mean color feature of test data
f = cifar_10_features(te_data(1:10000,:), N);
% Compute the normal distribution parameters
[MU, SIGMA, p] = cifar_10_bayes_learn_better(F(1:50000,:), tr_labels(1:50000,:));
% Returns the Bayesian optimal class
c = cifar_10_bayes_classify_better(f(1:10000,:), MU, SIGMA, p);
% Evaluate the prediction
cifar_10_evaluate(c, te_labels(1:10000));

%% Accuracy plot

window_size = [32, 16, 8, 4];
accuracy = [24.58, 36.6, 33.7, 31.09];
plot(window_size, accuracy, '*-');
grid on
xlabel('Sub-window size')
ylabel('Accuracy')
title('Bayesian with extended features')