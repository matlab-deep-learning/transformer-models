%% LanguageModelingWithFinBERT
% The FinBERT sentiment analysis model was fine-tuned from a FinBERT
% language model. The FinBERT language model is a BERT model that has been 
% trained on financial text data. We can interact with the FinBERT language
% model just as with BERT.

%% Construct models
% We will compare the BERT and FinBERT language models.
finbert_mdl = finbert('Model','language-model');
bert_mdl = bert();

%% Predict masked token
% Create an input string with a [MASK] token and call |predictMaskedToken|.
str = "My portfolio is heavily invested in stocks.";
maskedStr = strrep(str,"stocks","[MASK]")
bertPrediction = predictMaskedToken(bert_mdl,maskedStr)
finbertPrediction = predictMaskedToken(finbert_mdl,maskedStr)

%% Use the language model directly
% The |predictMaskedToken| function does not return probabilities across
% the tokenizer's vocabulary. Further it tokenizes inputs for the user, so
% it would be inappropriate to use it for training a language model. Below
% we demonstrate how to interact with the |languageModel| functions
% directly.

%% Tokenize
% The FinBERT and default BERT tokenizers are identical, we only need to
% tokenize once.
tokens = finbert_mdl.Tokenizer.tokenize(str);
tokens{1}

%% Mask a token
% Lets mask the "stocks" token. Note that we have to account for the [SEP]
% token that the |BERTTokenizer| appended.
maskedIdx = size(tokens{1},2)-2;
tokens{1}(maskedIdx) = finbert_mdl.Tokenizer.MaskToken;

%% Encode the tokens
% Note that the output is a cell of sequences - for workflows using 
% batches of observations this must be padded with |padsequences|. Here we
% can just use the single observation.
x = finbert_mdl.Tokenizer.encodeTokens(tokens);
x = x{1};

%% Call language model
% Call the language model of each model to get probabilities across the 
% vocabulary.
finbert_probabilities = finbert.languageModel(x,finbert_mdl.Parameters);
finbert_maskedTokenProbabilities = finbert_probabilities(:,maskedIdx);
bert_probabilities = bert.languageModel(x,bert_mdl.Parameters);
bert_maskedTokenProbabilites = bert_probabilities(:,maskedIdx);

%% Sample the probability
% Lets sample and decode the 10 most likely tokens.
numSamples = 10;
[~,finbert_sampleIdx] = maxk(extractdata(finbert_maskedTokenProbabilities),numSamples);
[~,bert_sampleIdx] = maxk(extractdata(bert_maskedTokenProbabilites),numSamples);

% The tokenizer lets us decode the predicted token:
bert_predictedTokens = arrayfun(@(idx) finbert_mdl.Tokenizer.decode(idx), bert_sampleIdx)
finbert_predictedTokens = arrayfun(@(idx) finbert_mdl.Tokenizer.decode(idx), finbert_sampleIdx)