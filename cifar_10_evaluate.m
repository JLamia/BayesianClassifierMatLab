function cifar_10_evaluate(pred, gt)

indexes = find(pred==gt);
l = length(indexes);
p = l/length(gt)*100;
fprintf('The classification accuracy is %f %%\n', p);

end