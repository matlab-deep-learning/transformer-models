%% Pretrained Document Embeddings with BERT
% This example shows how to use the BERT model as a contextual document
% embedding, then train a simple model on those embedding vectors. This is
% a quick and less memory-intensive way to use BERT compared to the
% fine-tuning example ClassifyTextDataUsingBERT.m

%% Import Data
% Import the factory reports data and extract the string data and classes
% we wish to train a classifier on.
filename = "factoryReports.csv";
data = readtable(filename,'TextType','string');
targets = categorical(data.Category);
predictorStrings = data.Description;

%% Tokenize, encode and embed with BERT
% The predictorStrings have to be tokenized and encoded before BERT can be
% used.

% Load a BERT model - here we use one of the smaller models which is less
% powerful as it has less parameters, however that makes it less memory
% intensive and quicker to use.
mdl = bert('Model','small');

% The helper function bertEmbed shows how to use this model and its
% tokenizer to embed a collection of documents.
predictor = bertEmbed(mdl,predictorStrings);

%% Train a simple model on the embedded documents
% The array predictor is B-by-C where C is
% mdl.Parameters.Hyperparameters.HiddenSize and B is the number of
% observations. We can pass predictor to a model that
% classifies each observation as the corresponding entry in the categorical
% targets.

% Data with no spatial dimensions can be passed into a featureInputLayer
% for use in a neural network.
hiddenSize = 50;
numClasses = numel(categories(targets));
layers = [
    featureInputLayer(size(predictor,2),'Normalization','zscore')
    fullyConnectedLayer(hiddenSize)
    layerNormalizationLayer()
    reluLayer()
    fullyConnectedLayer(numClasses)
    softmaxLayer()
    classificationLayer()];
opts = trainingOptions('adam',...
    'MaxEpochs',10,...
    'MiniBatchSize',32,...
    'Shuffle','every-epoch',...
    'Plots','training-progress');

net = trainNetwork(predictor,targets,layers,opts);

%% Predict on new data
% To use net for new data we have to repeat the earlier steps using BERT as
% an embedding. 
newReport = "Coolant is pooling underneath sorter.";
newEmbedding = bertEmbed(mdl,newReport);
net.classify(newEmbedding)

%% Supporting Functions
function embedded = bertEmbed(mdl,predictorStrings)
% Embed a string str using the BERT model mdl.
predictorSequences = mdl.Tokenizer.encode(predictorStrings);
% First ensure the sequences are not longer than the BERT model can handle,
% then pad.
predictorSequencesTruncated = truncateSequences(predictorSequences,mdl.Parameters.Hyperparameters.NumContext);
x = padsequences(predictorSequencesTruncated,2,'PaddingValue',mdl.Tokenizer.PaddingCode);

% Now embed using bert.model - note if this step is too memory intensive
% you can operate on minibatches via minibatchqueue and cache the results
% as you go.
embeddedSequence = bert.model(x,mdl.Parameters);

% Now we want to remove the sequence dimension - this is sometimes
% called "pooling". By virtue of the self-attention mechanism, each
% sequence element is able to attend to each other sequence elemebt in
% bert.model. This motivates the following simple pooling - just take the
% first token.
embedded = squeeze(embeddedSequence(:,1,:));

% We have to extract the data out of the dlarray wrapper.
embedded = extractdata(embedded);

% Finally we will pass this data to a featureInputLayer, which requires
% that each row of embedded is an observation, rather than each column as
% in its current form. Simply transpose the data.
embedded = embedded.';
end