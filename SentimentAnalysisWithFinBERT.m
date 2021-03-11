%% Analyze Sentiment Using FinBERT
% This example shows how to analyze sentiment of financial data using a
% pretrained FinBERT model.
%
% FinBERT is a sentiment analysis model trained on financial text data and
% fine-tuned for sentiment analysis. The model is based on the BERT-Base
% architecture.
%
% This example shows how to classify the sentiment of financial news
% reports using a pretrained FinBERT model.

%% Load Pretrained FinBERT Model
% Load a pretrained FinBERT model using the |finbert| function. The model
% consists of a tokenizer that encodes text as sequences of integers, and
% a structure of parameters.
mdl = finbert

%%
% View the FinBERT model tokenizer. The tokenizer encodes text as sequences
% of integers and holds the details of padding, start, separator and mask
% tokens.
tokenizer = mdl.Tokenizer

%% Analyze Sentiment in Text
% Create an string array of text data.
str = [
    "In an unprecendented move the stock market has hit new records today following news of a new vaccine."
    "Businesses in this sector suffer dramatic losses on the back of the pandemic."
    "The ship unloader is totally enclosed along the entire conveying line to the storage facilities."
    "Experts estimate the value of its remaining stake in the company at $ 27 million."
    "The company said that sales in the three months to the end of March slid to EUR86 .4 m US$ 113.4 m from EUR91 .2 m last year."
    "Finance experts calculate that it has lost EUR 4mn in the failed project."
    "They signed a large deal with an international industrial group which will deliver automation solutions and connectivity services."
    "Operating profit rose to EUR 5mn from EUR 2.8 mn in the fourth quarter of 2008"];

%%
% Encode the text data as a sequence of tokens using the FinBERT model
% tokenizer.
tokens = encode(tokenizer,str);

%% 
% Pad the sequences to have the same length using the |padsequences|
% function. Specify the padding value to match the FinBERT model tokenizer.
X = padsequences(tokens,2,"PaddingValue",mdl.Tokenizer.PaddingCode);

%% 
% Evaluate the sentiment and sentiment scores using the
% |finbert.sentimentModel| function.
[sentiment,scores] = finbert.sentimentModel(X,mdl.Parameters)
