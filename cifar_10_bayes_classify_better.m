function c = cifar_10_bayes_classify_better(f, MU, SIGMA, p)

% We have 10 classes from 0 to 9
number_of_classes = 10;
% 10000 training data x 10 values
posteriori = zeros(size(f, 1), 10);

% for i = 1:size(f, 1)
%     for j = 1:number_of_classes
%         posteriori(i,j) = mvnpdf(f(i,:), MU(j,:), SIGMA{j,1})*p(j);
%     end
% end

for i = 1:size(f, 1)
    for j = 1:number_of_classes
        posteriori(i,j) = mvnpdf(f(i,:), MU(j,:), SIGMA{j,1})*p(j);
    end
end

[val,loc] = max(posteriori');
c = loc' - 1;

end