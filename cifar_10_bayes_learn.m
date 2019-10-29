function [mu, sigma, p] = cifar_10_bayes_learn(F, tr_labels)

% We have 10 classes form 0 to 9
number_of_classes = 10;
mu = zeros(number_of_classes, 3);
sigma = zeros(number_of_classes, 3);
p = zeros(number_of_classes, 1);

for i = 1:number_of_classes
    %indices = find(tr_labels == i);
    indices = tr_labels == i-1;
    features = F(indices,:);
    mu(i,:) = mean(features);
    sigma(i,:) = std(features);
    p(i,:) = sum(indices == 1)/size(F, 1);
end

end