function mdl = finbert(nvp)
% finbert   Pretrained FinBERT transformer model
%   mdl = finbert loads a pretrained FinBERT model and downloads the model
%   weights and vocab file if necessary.
%
%   mdl = finbert('Model', modelName) loads the FinBERT model specified by
%   modelName. Supported values for modelName are "sentiment-model"
%   (default) and "language-model".

% Copyright 2021 The MathWorks, Inc.
arguments
    nvp.Model (1,1) string {mustBeMember(nvp.Model,...
        ["language-model";
        "sentiment-model"])} = "sentiment-model"
end
% Download the license file
finbert.internal.getSupportFilePath(nvp.Model,'finbert.RIGHTS');
params = finbert.load(nvp.Model);
% Get the IgnoreCase hyperparameter, then remove it, downstream code
% shouldn't need it.
ignoreCase = params.Hyperparameters.IgnoreCase;
% Get vocab file path
vocabFile = finbert.internal.getSupportFilePath(nvp.Model,"vocab.txt");
params.Hyperparameters = rmfield(params.Hyperparameters,'IgnoreCase');
mdl = struct(...
    'Tokenizer',bert.tokenizer.BERTTokenizer(vocabFile,'IgnoreCase',ignoreCase),...
    'Parameters',params);
end