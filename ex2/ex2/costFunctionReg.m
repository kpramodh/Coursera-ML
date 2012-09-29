function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
a=X*theta;
sum1 =0; sum2=0;
for i=1:m,
  sum1 = sum1 + (-y(i)*log(sigmoid(a(i))) - (1-y(i))*log(1-sigmoid(a(i))));
end;

for j=2:n
  sum2 = sum2 + theta(j)^2;
 end;
 
J = (1/m)*sum1 + (lambda/(2*m))*sum2;

grad = (1/m)*(X'*(sigmoid(a)-y))+(lambda/m)*theta;

grad(1) = grad(1) - (lambda/m)*theta(1);





% =============================================================

end
