function Z = normalization(X, g, b)
% normalization   Layer Normalization
%
%   Z = normalization(X, g, b) applies layer normalization to the input X.
%   Layer normzalization is described in [1].
%
%   Inputs:
%       X - A numFeatures-by-numInputSubwords-by-numObs input array.
%       g - A numFeatures-by-1 weight vector.
%       b - A numFeatures-by-1 bias vector.
%
%   Outputs:
%       Z - A numFeatures-by-numInputSubwords-by-numObs output array.
%
%   References:
%
%   [1] Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer
%       Normalization", https://arxiv.org/abs/1607.06450

normalizationDimension = 1;

epsilon = single(1e-5);

U = mean(X, normalizationDimension);
S = mean((X-U).^2, normalizationDimension);
X = (X-U) ./ sqrt(S + epsilon);
Z = g.*X + b;

end