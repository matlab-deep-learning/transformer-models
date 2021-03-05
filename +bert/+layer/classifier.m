function y = classifier(x,p)
% classifier   The standard BERT classifier, a single fullyconnect.
%
%   Z = classifier(X,classifierWeights) applies a fullyconnect operation to
%   the input X with weights classifierWeights.kernel and bias
%   classifierWeights.bias. The input X must be an unformatted dlarray of
%   size hiddenSize-by-numObs. The classifierWeights.kernel must be of size
%   outputSize-by-hiddenSize, and the classifierWeights.bias must be of
%   size outputSize-by-1.

% Copyright 2021 The MathWorks, Inc.
y = transformer.layer.convolution1d(x,p.kernel,p.bias);
end