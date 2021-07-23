function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];  % add column of ones to X
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

h1 = sigmoid(X * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
hypothesis = h2;
% Each row of h2 is a real number in the sigmoid range [0-1] such that the 
% position of the largest of those 10 numbers gives the predicted digit. 
yVec = zeros(num_labels, 1);
yVec(y(1)) = 1;

% Calculate the (unregularized) cost.
for i = 1 : m
  yVec = zeros(num_labels, 1);
  yVec(y(i)) = 1;
  % Garry: the following (clean) line vectorizes the messier "for k" loop below.
  J += (-yVec' * log(hypothesis(i, :)')) - ((1 .- yVec') * log(1 .- hypothesis(i, :)'));

  % Note: the following loop does the trick, iterating over each label. But see
  % the vectorized implementation above.
##  for k = 1 : num_labels
##    yVecValue = yVec(k); 
##    predictedValue = h(1, k); 
##    J += (-yVecValue * (log(predictedValue))) - ((1 - yVecValue) * log(1 - predictedValue)); 
##  endfor
endfor

% Calculate the COST regularization term.
noBiasTheta1 = Theta1(:, 2:columns(Theta1)); 
sumTheta1 = sum(noBiasTheta1(:) .^ 2); 
noBiasTheta2 = Theta2(:, 2:columns(Theta2)); 
sumTheta2 = sum(noBiasTheta2(:) .^ 2); 
regularizationCost = (lambda / (2 * m)) * (sumTheta1 + sumTheta2);

% Calculate the THETA regularization terms. Do the calculation without the bias
% column and then add in a column of zeros so the resulting regularization 
% matrix is size-compatible with Theta.
Theta1_regularization = [zeros(rows(Theta1),1) ((lambda / m) .* Theta1(:, 2:end))];
Theta2_regularization = [zeros(rows(Theta2),1) ((lambda / m) .* Theta2(:, 2:end))];
assert(size(Theta1_regularization) == size(Theta1));
assert(size(Theta2_regularization) == size(Theta2));

% Calculate the regularized cost.
J = ((1 / m) * J) + regularizationCost;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1 : m 
  
  % 1. Perform a feedforward pass. 
  a1 = X(t, :);
  z2 = Theta1 * a1';
  a2 = sigmoid(z2);
  
  a2 = [1; a2];
  z3 = Theta2 * a2;
  h_x = a3 = sigmoid(z3);

  % 2. The current training example belongs to class k.
  yVec = zeros(num_labels, 1);
  yVec(y(t)) = 1;
  d3 = a3 .- yVec;
  
  % 3. For the hidden layer l = 2.
  z2 = [1; z2];  % add back the bias node for z2
  d2 = (Theta2' * d3) .* sigmoidGradient(z2);
  d2 = d2(2:end);  % remove d2(0)

  % 4. Accumulate the gradient from this example.
  Theta2_grad = Theta2_grad + (d3 * a2');
  Theta1_grad = Theta1_grad + (d2 * a1);

endfor

% 5. Obtain the unregularized gradient by dividing by 1/m.
Theta2_grad = (1 / m) .* Theta2_grad;
Theta1_grad = (1 / m) .* Theta1_grad;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad = Theta2_grad .+ Theta2_regularization;
Theta1_grad = Theta1_grad .+ Theta1_regularization;

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
