function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
a = h - y;
a= a .^ 2;
%calculate regularization term
t = theta;

t(1,:) = [];

t = t .^2;  

J = (1/(2*m)) * sum(a) + (lambda/(2*m))*sum(t);


n = size(theta,1);

grad(1,1) = (1/m)*sum((h-y).*X(:,1)); 
for i=2:n,
grad(i,1) = (1/m)*sum((h-y).*X(:,i)) + (lambda/m)*theta(i,1);
end;











% =========================================================================

grad = grad(:);

end
