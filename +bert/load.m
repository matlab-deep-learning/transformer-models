function params = load(modelName)
% load   Load BERT parameters
%
%   parameters = load(modelName) will load the model weights associated to
%   modelName and the hyperparameters associated to that model.

% Copyright 2021 The MathWorks, Inc.
arguments
    modelName (1,1) string
end

% the URLs don't have bert- at the start.
paramsStructFile = bert.internal.getSupportFilePath(modelName,"parameters.mat");
paramsStruct = load(paramsStructFile);

params = struct(...
    'Hyperparameters',paramsStruct.Hyperparameters,...
    'Weights',bert.internal.createParameterStruct(paramsStruct.Weights));
end