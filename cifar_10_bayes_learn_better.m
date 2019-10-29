function [MU, SIGMA, p] = cifar_10_bayes_learn_better(F, tr_labels)

% We have 10 classes form 0 to 9
number_of_classes = 10;
MU = zeros(number_of_classes, size(F, 2));
SIGMA = cell(number_of_classes, 1);
p = zeros(number_of_classes, 1);

for i = 1:number_of_classes
    %indices = find(tr_labels == i);
    indices = tr_labels == i-1;
    features = F(indices,:);
    MU(i,:) = mean(features);
    SIGMA{i,1} = cov(features); % FIXME: do we need REAL??
    p(i,:) = sum(indices == 1)/size(F, 1);
end

end