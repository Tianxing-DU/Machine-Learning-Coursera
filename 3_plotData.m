function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
% Using for loop and if condition, however legend cannot be plotted properly. 
% Considering I will not use it for my research visualisation, so here is just okay.
% 我的感悟是，必须要plot 两个data两个独立的命令，legend才会分别赋值

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 5);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 5);

% 我使用for loop + if condition 也是可以完成的（可惜legend不对，这里于是没有保存）



% =========================================================================



hold off;

end
