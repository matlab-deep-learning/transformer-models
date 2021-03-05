%% BERT as a Language Model
% One of the pre-training tasks for the BERT model is masked language modeling.
% The language model task is to replace a token in the input sequence with
% a mask token [MASK] and train the model to predict the original token.
% The language model can be used via the bert.languageModel function.

%% Construct a BERT model and tokenize an input sentence
mdl = bert();

%% Predict a masked token
% If you want to simply replace a masked token with the token that the
% model thinks is most likely, the |predictMaskedToken| function can be
% used.
sentence = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.";
maskedSentence = strrep(sentence,"processing","[MASK]")
predictedSentence = predictMaskedToken(mdl,maskedSentence)

%% Using the language model directly
% The |predictMaskedToken| function is convenient but does not return the
% probabilities of the masked token across the entire vocabulary, and
% couldn't be used for training your own language model.
% To interact with the language model on a lower level use the
% |bert.languageModel| function.

%% Tokenize
% First tokenize the string into sub-strings.
% Notice that the |BERTTokenizer| prepends a [CLS] token and appends a
% [SEP] token.
tokens = mdl.Tokenizer.tokenize(sentence);

%% Mask an input token
% Lets mask the 3rd token corresponding to "language", then encode the
% tokens.
toMask = 3;
tokens{1}(toMask) = mdl.Tokenizer.MaskToken;
X = mdl.Tokenizer.encodeTokens(tokens);
X = X{1};

%% Call the language model
% The |bert.languageModel| returns a probability distribution across the
% tokenizer's vocabulary for each token in the input. The language model
% outputs are of size VocabularySize x SequenceLength.
% Here choose only the probability corresponding to the masked token.
probs = bert.languageModel(X,mdl.Parameters);
languageModelOutputSize = size(probs)
maskTokenProbs = probs(:,toMask);

%% Sample the probability
% Finally lets see what the top 10 predictions are by the model.
[~,topTen] = maxk(extractdata(maskTokenProbs),10);
for i = 1:10
    disp("Predicted Token:"+mdl.Tokenizer.decode(topTen(i)));
    disp("Probability:"+extractdata(maskTokenProbs(topTen(i))));
end