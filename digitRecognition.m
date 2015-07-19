function g = sigmoid(z)
  % SIGMOID Compute sigmoid function
  g = 1.0 ./ (1.0 + exp(-z));
end

function p = predict(Theta1, Theta2, X)
  %PREDICT Predict the label of an input given a trained neural network
  %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  %   trained weights of a neural network (Theta1, Theta2)
  
  % Useful values
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);
  
  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  
  foo = sigmoid(X * Theta1');
  foo = [ones(size(foo, 1), 1) foo ];
  
  [foo,p] = max(sigmoid(foo * Theta2'), [], 2);

end


function p = predictOneVsAll(all_theta, X)
  %PREDICT Predict the label for a trained one-vs-all classifier. The labels 
  %are in the range 1..K, where K = size(all_theta, 1). 
  %  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
  %  for each example in the matrix X. Note that X contains the examples in
  %  rows. all_theta is a matrix where the i-th row is a trained logistic
  %  regression theta vector for the i-th class. 
  
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  p = zeros(size(X, 1), 1);
  
  % Add ones to the X data matrix
  X = [ones(m, 1) X];

  [foo,p] = max(sigmoid(X * all_theta'), [], 2);

end


function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 
  
  % Initialize some useful values
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  
  
  h = sigmoid(X * theta);
  theta1 = theta(2:end);
  
  J =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h)) + lambda/(2*m)*sum(theta1'*theta1);
  
  grad = (1/m).*X'*(h-y);
  grad(2:end) = grad(2:end) + (lambda/m*theta1);
  grad = grad(:);

end


function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
  %   logisitc regression classifiers and returns each of these classifiers
  %   in a matrix all_theta, where the i-th row of all_theta corresponds 
  %   to the classifier for label i
  
  % Some useful variables
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_labels, n + 1);
  
  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  
  for c = 1:num_labels
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = ...
              fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                      initial_theta, options);
    all_theta(c,:) = theta;
  end

end


function [h, display_array] = displayData(X, example_width)
    %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    %   stored in X in a nice grid. It returns the figure handle h and the 
    %   displayed array if requested.
    
    % Set example_width automatically if not passed in
    if ~exist('example_width', 'var') || isempty(example_width) 
    	example_width = round(sqrt(size(X, 2)));
    end
    
    % Gray Image
    colormap(gray);
    
    % Compute rows, cols
    [m n] = size(X);
    example_height = (n / example_width);
    
    % Compute number of items to display
    display_rows = floor(sqrt(m));
    display_cols = ceil(m / display_rows);
    
    % Between images padding
    pad = 1;
    
    % Setup blank display
    display_array = - ones(pad + display_rows * (example_height + pad), ...
                           pad + display_cols * (example_width + pad));
    
    % Copy each example into a patch on the display array
    curr_ex = 1;
    for j = 1:display_rows
      for i = 1:display_cols
      if (curr_ex > m)
           break; 
        end
        % Copy the patch
    		
        % Get the max value of the patch
        max_val = max(abs(X(curr_ex, :)));
        display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), pad + (i - 1) * (example_width + pad) + (1:example_width)) = reshape(X(curr_ex, :), example_height, example_width) / max_val;
        curr_ex = curr_ex + 1;
      end
        if (curr_ex > m)
         break; 
      end
    end
    
    % Display Image
    h = imagesc(display_array, [-1 1]);
    
    % Do not show axis
    axis image off
    
    drawnow;

end


%% Machine Learning : One-vs-all

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n');

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



sel = X(pred!=y, :);

displayData(sel);

fprintf('Unrecognized patterns.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;



%% Machine Learning : Neural Networks


%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Pameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');

%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;



sel = X(pred!=y, :);

displayData(sel);

fprintf('Unrecognized patterns.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;




%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

