function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Predicted cost H(x)
h_x = sigmoid(X*theta);

% theta after jumping the first one (which does not need to be regularised
theta_j1 = theta(2:n);
% 如果不使用n也可以用end代替：theta1 = [0 ; theta(2:end, :)];

J = (((-y)'*log(h_x)-(1-y)'*log(1-h_x))/m) + (lambda/(2*m)) * theta_j1'*theta_j1; 
% grad calculated by the theta 1 (without regularisation)
grad = 1/m * X'*(h_x - y);

% replace the value in grad after doing regularisation
grad(2:n)= grad(2:n)+theta_j1*lambda/m;

% 上面是我原来的方法，使用了替代2:n的grad，但是也可以使用它的提示方式
% grad =1/m * X' (h_x -y); % 仍然用原来的theta（1号位没有被取代）
% temp= theta;
% temp(1) = 0; % 取代1号位，这样在算regularizing的时候，一号位得到的grad加成=0，相当于没有改变
% grad = grad+temp *lambda/m;









% =============================================================

grad = grad(:);

end
