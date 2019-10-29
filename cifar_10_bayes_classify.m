function c = cifar_10_bayes_classify(f, mu, sigma, p)

% We have 10 classes from 0 to 9
number_of_classes = 10;
rows = size(f, 1);
% 10000 training data x 10 values for each class
posteriori = zeros(size(f, 1), 10);

for i = 1:rows
    for j = 1:number_of_classes
        posteriori(i,j) = normpdf(f(i,1), mu(j,1), sigma(j,1))*...
            normpdf(f(i,2), mu(j,2), sigma(j,2))*...
            normpdf(f(i,3), mu(j,3), sigma(j,3))*p(j);
    end
end
[val,loc] = max(posteriori');
c = loc' - 1;

end