function mdl = bert(nvp)
% bert   Pretrained BERT transformer model
%   mdl = bert loads a pretrained BERT-Base model and downloads the model
%   weights and vocab file if necessary.
%
%   mdl = bert('Model', modelName) loads the BERT model specified by
%   modelName.  Supported values for modelName are "base" (default),
%   "multilingual-cased","medium","small","mini", and "tiny".

% Copyright 2021 The MathWorks, Inc.
arguments
    nvp.Model (1,1) string {mustBeMember(nvp.Model,[
        "base"
        "multilingual-cased"
        "medium"
        "small"
        "mini"        
        "tiny"])} = "base"
end
% Download the license file
bert.internal.getSupportFilePath(nvp.Model,"bert.RIGHTS");
params = bert.load(nvp.Model);
% Get the IgnoreCase hyperparameter, then remove it, downstream code
% shouldn't need it.
ignoreCase = params.Hyperparameters.IgnoreCase;
% Get vocab file
vocabFile = bert.internal.getSupportFilePath(nvp.Model,"vocab.txt");
params.Hyperparameters = rmfield(params.Hyperparameters,'IgnoreCase');
mdl = struct(...
    'Tokenizer',bert.tokenizer.BERTTokenizer(vocabFile,'IgnoreCase',ignoreCase),...
    'Parameters',params);
end