function z = languageModel(x,p)
% languageModel   The BERT language model.
%
%   Z = bert.languageModel(X,parameters) performs inference with a BERT model
%   on the input X, and applies the output layer projection onto the
%   associated vocabulary. The input X is a 1-by-numInputTokens-by-numObs
%   array of encoded tokens. The return is an array Z of size
%   vocabularySize-by-numInputTokens-by-numObs. In particular the language model is
%   trained to predict a reasonable word for each masked input token.

% Copyright 2021 The MathWorks, Inc.
if ~isfield(p.Weights,'masked_LM')
    error("bert:languageModel:MissingLMWeights","Parameters do not include masked_LM weights");
end
z = bert.model(x,p);
z = bert.layer.languageModelHead(z,p.Weights.masked_LM,p.Weights.embeddings.word_embeddings);
end