function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); %number of parameters
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Predicted cost H(x)
h_x = sigmoid(X*theta);

% theta after jumping the first one (which does not need to be regularised
theta_j1 = theta(2:n);

J = (((-y)'*log(h_x)-(1-y)'*log(1-h_x))/m) + (lambda/(2*m)) * theta_j1'*theta_j1; % 之前这里有过错误，(2*m)这里的括号问题。以后做数学运算的时候值得注意
% grad calculated by the theta 1 (without regularisation)
grad = 1/m * X'*(h_x - y);

% replace the value in grad after doing regularisation
grad(2:n)= grad(2:n)+theta_j1*lambda/m;


% =============================================================

end
