function f = cifar_10_features(trdata, N)

assert(N==32 | N==16 | N==8 | N==4, 'N must be 32, 16, 8 or 4')

rows = size(trdata, 1);
n_images = (32/N)^2*3;
f = zeros(rows, n_images);

% for i = 1:rows
%     data_sample = trdata(i,:);
%     f_r = mean(data_sample(1:1024));
%     f_g = mean(data_sample(1025:2048));
%     f_b = mean(data_sample(2049:3072));
%     f(i,:) = [f_r, f_g, f_b];
% end

for i = 1:rows
    data_sample = trdata(i,:);
    for j = 1:n_images
        f(i,j) =...
            mean(data_sample(1+(j-1)*3072/n_images:j*3072/n_images));
    end
end

end