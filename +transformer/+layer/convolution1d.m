function Z = convolution1d(X, W, b)
% convolution1d   A fully connected layer
%
%   Z = convolution1d(X, W, b) applies a fully connected layer. We call it
%   1-D convolution because this is what it's called in the original GPT-2
%   repo.
%
%   Inputs:
%       X   - A numInputFeatures-by-numInputSubwords array.
%       W   - A numOutputFeatures-by-numInputFeatures weight matrix.
%       b   - A numOutputFeatures-by-1 bias vector.
%
%   Output:
%       Z   - A numOutputFeatures-by-numInputSubwords array.

Z = dlmtimes(W,X) + b;

end