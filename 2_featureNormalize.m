function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X); % 都是给出columwise的mean 或std（默认）
sigma = std(X);

N = columns(X);% 这里也可以使用size(X,2), 2 在这里会给出column得多少， 1是row column
for i=1:N,
   Mu = mu(:,i);
   Sigma = sigma(:,i);
   X_norm (:, i) = (X(:,i) -Mu)/Sigma;
   i=i+1; % maybe I do not need it here
end
% 这里的问题是X_norm不会被 tempo memory记录


% =============_norm======== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
