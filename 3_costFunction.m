function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));% grad就是gradient，delta（J)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h_x = sigmoid(X*theta);
J = ((-y)'*log(h_x)-(1-y)'*log(1-h_x))/m;
% 这里也可以用sum，那么就需要transpose log（h_x),而不是y。
% transpose y的好处是把matrix变成了1X1， 而不是100X100，相当于直接求sum
% 换句话说，如果遇到求sum的，试一试能不能变成1X1的matrix

grad = 1/m * X'*(h_x - y);





% =============================================================

end
