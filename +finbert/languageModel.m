function z = languageModel(x,params)
% languageModel   The FinBERT language model.
%
%   Z = finbert.languageModel(X,parameters) performs inference with a FinBERT model
%   on the input X, and applies the output layer projection onto the
%   associated vocabulary. The input X is a 1-by-numInputTokens-by-numObs
%   array of encoded tokens. The return is an array Z of size
%   vocabularySize-by-numInputTokens-by-numObs. In particular the language model is
%   trained to predict a reasonable word for each masked input token.

% Copyright 2021 The MathWorks, Inc.
z = bert.languageModel(x,params);
end