function z = classifierHead(x,poolerWeights,classifierWeights)
% classifierHead   The standard classification head for a BERT model.
% 
%   Z = classifierHead(X,poolerWeights,classifierWeights) applies
%   bert.layer.pooler and bert.layer.classifier to X with poolerWeights and
%   classifierWeights respectively. Both poolerWeights and
%   classifierWeights must be structs with fields 'kernel' and 'bias'.

% Copyright 2021 The MathWorks, Inc.
z = bert.layer.pooler(x,poolerWeights);
z = bert.layer.classifier(z,classifierWeights);
end