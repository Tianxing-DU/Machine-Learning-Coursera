function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
    theta= theta-alpha/m*(X'*(X*theta -y));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    % theta_1 = theta(1) - alpha * (1/m) * sum((X*theta-y).*X(:,1));
    % theta_2 = theta(2) - alpha * (1/m) * sum((X*theta-y).*X(:,2));
    % Afterwards you are setting the temporary thetas (here called theta_1 and 
    % theta_2)correctly back to the "real" theta.
    % 上面的只适用于两个theta的情况                     







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
