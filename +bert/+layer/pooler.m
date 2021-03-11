function y = pooler(x,p)
% pooler   The standard BERT pooler. Takes first sequence element then 
% applies fullyconnect with tanh activation.
%
%   Z = pooler(X,poolerWeights) pools the input X using poolerWeights. The
%   input X must be a hiddenSize-by-numInputTokens-by-numObs unformatted
%   dlarray, such as the output of the bert.model function. The
%   poolerWeights must be a struct with fields 'kernel' and 'bias' of size
%   outputSize-by-hiddenSize and outputSize-by-1 respectively.

% Copyright 2021 The MathWorks, Inc.
z = squeeze(x(:,1,:));
y = transformer.layer.convolution1d(z,p.kernel,p.bias);
y = tanh(y);
end