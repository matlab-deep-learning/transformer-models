%% Predict Masked Tokens Using FinBERT
% This example shows how to predict masked tokens using a pretrained
% FinBERT model.
%
% BERT models are trained to perform various tasks. One of the tasks is
% known as masked language modeling which is the task of predicting tokens
% in text that have been replaced by a mask value.
%
% This example shows how to predict masked tokens for financial text data
% and calculate the token probabilities using a pretrained FinBERT model.

%% Load Pretrained FinBERT Model
% Load a pretrained FinBERT model using the |finbert| function. For
% language model workflows, set the "Model" option to "language-model". The
% model consists of a tokenizer that encodes text as sequences of integers,
% and a structure of parameters.
mdl = finbert("Model","language-model")

%%
% View the FinBERT model tokenizer. The tokenizer encodes text as sequences
% of integers and holds the details of padding, start, separator and mask
% tokens.
tokenizer = mdl.Tokenizer

%% Predict Masked Token
% Create a string containing a piece of text and replace a single word with
% the tokenizer mask token.
str = "Experts estimate the value of its remaining stake in the company at $ 27 million.";
strMasked = replace(str,"stake",tokenizer.MaskToken)

%%
% Predict the masked token using the |predictMaskedToken| function. The
% function returns the original string with the mask tokens replaced.
sentencePred = predictMaskedToken(mdl,strMasked)

%% Calculate Prediction Scores
% To get the prediction scores for each word in the model vocabulary, you
% can evaluate the language model directly using the |bert.languageModel|
% function.

%%
% First, tokenize the input sentence with the FinBERT model tokenizer using
% the |tokenize| function. Note that the tokenizer may split single words
% and also prepends a [CLS] token and appends a [SEP] token to the input.
tokens = tokenize(tokenizer,str);
tokens{1}

%%
% Replace one of the tokens with the mask token.
idx = 9;
tokens{1}(idx) = tokenizer.MaskToken

%%
% Encode the tokens using the FinBERT model tokenizer using the
% |encodeTokens| function.
X = encodeTokens(tokenizer,tokens);
X{1}

%%
% To get the predictions scores from for the encoded tokens, evaluate the
% FinBERT language model directly using the |bert.languageModel| function.
% The language model output is a VocabularySize-by-SequenceLength array.
scores = bert.languageModel(X{1},mdl.Parameters);

%%
% View the tokens of the FinBERT model vocabulary corresponding to the top
% 10 prediction scores for the mask token.
[~,idxTop] = maxk(extractdata(scores(:,idx)),10);
tbl = table;
tbl.Token = arrayfun(@(x) decode(tokenizer,x), idxTop);
tbl.Score = scores(idxTop,idx)
