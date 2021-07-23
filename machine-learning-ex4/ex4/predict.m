function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 

## What's listed here is the course-provided implementation. I've elected to use
## my implementation below which seems functionally equivalent, if not quite as
## elegant - Garry.

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

% Provided implementation ends above; my implementation begins below:

% ====== Layer 2 computation ======
X = [ones(m, 1) X];  % Add ones to the data matrix
z2 = Theta1 * X';
a2 = sigmoid(z2);

% ====== Layer 3 computation ======
a2 = [ones(m, 1), a2'];  % Add ones to the data matrix
z3 = Theta2 * a2';
a3 = sigmoid(z3);

[x, p] = max(a3', [], 2);
assert(size(p, 1) == m);

% =========================================================================

end
