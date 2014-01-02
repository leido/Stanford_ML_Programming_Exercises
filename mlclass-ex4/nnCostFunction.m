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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a_1 = X;
X = [ones(m,1) X]; %5000 x 401
Z_2 = X*Theta1'; %5000 x 25
a_2 = sigmoid(Z_2); %5000 x 25
a_2 = [ones(m,1) a_2]; % 5000 x 26
Z_3 = a_2*Theta2'; % 5000 x 10
h_x = sigmoid(Z_3); % 5000 x 10
for k=1:m
    temp_y = 1:num_labels;
    temp_y = temp_y';
    yvec = temp_y==y(k);
    J = J - log(h_x(k,:))*yvec-(log(1-h_x(k,:))*(1-yvec));
end
J=J/m;
reg_terms = 0;
for j=1:hidden_layer_size
    for k=1:input_layer_size
        reg_terms = reg_terms + Theta1(j,k+1)^2;
    end
end
for j = 1:num_labels
    for k=1:hidden_layer_size
        reg_terms = reg_terms + Theta2(j,k+1)^2;
    end
end
reg_terms = reg_terms*lambda/2/m;
J = J + reg_terms;
temp = zeros(m,num_labels);
for i=1:m
    temp(i,y(i))=1;
end
delta_3 = h_x - temp; % 5000 x 10
delta_2 = delta_3*Theta2(:,2:end).*sigmoidGradient(Z_2); % 5000 x 25

t1 = lambda.*Theta1;
t1(:,1) = 0;
Theta1_grad = (delta_2'*X+t1)./m;

t2 = lambda.*Theta2;
t2(:,1) = 0;
Theta2_grad = (delta_3'*a_2+t2)./m;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
