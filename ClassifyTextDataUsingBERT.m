%% Classify Text Data Using BERT
% This example shows how to classify text data using a pretrained BERT
% model as a feature extractor.
%
% The simplest use of a pretrained BERT model is to use it as a feature
% extractor. In particular, you can use the BERT model to convert documents
% to feature vectors which you can then use as input to train a deep
% learning classification network.
%
% This example shows how to use a pretrained BERT model to classify failure
% events given a data set of factory reports.

%% Load Pretrained BERT Model
% Load a pretrained BERT model using the |bert| function. The model
% consists of a tokenizer that encodes text as sequences of integers, and
% a structure of parameters.
mdl = bert

%%
% View the BERT model tokenizer. The tokenizer encodes text as sequences of
% integers and holds the details of padding, start, separator and mask
% tokens.
tokenizer = mdl.Tokenizer

%% Load Data
% Load the example data. The file |factoryReports.csv| contains factory
% reports, including a text description and categorical labels for each
% event.

filename = "factoryReports.csv";
data = readtable(filename,"TextType","string");
head(data)

%%
% The goal of this example is to classify events by the label in the
% |Category| column. To divide the data into classes, convert these labels
% to categorical.
data.Category = categorical(data.Category);

%%
% View the number of classes.
classes = categories(data.Category);
numClasses = numel(classes)

%%
% View the distribution of the classes in the data using a histogram.
figure
histogram(data.Category);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

%%
% Encode the text data using the BERT model tokenizer using the |encode|
% function and add the tokens to the training data table.
data.Tokens = encode(tokenizer, data.Description);

%%
% The next step is to partition it into sets for training and validation.
% Partition the data into a training partition and a held-out partition for
% validation and testing. Specify the holdout percentage to be 20%.
cvp = cvpartition(data.Category,"Holdout",0.2);
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);

%%
% View the number of training and validation observations.
numObservationsTrain = size(dataTrain,1)
numObservationsValidation = size(dataValidation,1)

%%
% Extract the text data, labels, and encoded BERT tokens from the
% partitioned tables.
textDataTrain = dataTrain.Description;
textDataValidation = dataValidation.Description;

TTrain = dataTrain.Category;
TValidation = dataValidation.Category;

tokensTrain = dataTrain.Tokens;
tokensValidation = dataValidation.Tokens;

%%
% To check that you have imported the data correctly, visualize the
% training text data using a word cloud.

figure
wordcloud(textDataTrain);
title("Training Data")

%%
% View the BERT token codes of the first few training documents.
tokensTrain{1:5}

%% Prepare Data for Training
% Convert the documents to feature vectors using the BERT model as a
% feature extractor.

% To extract the features of the training data by iterating over
% mini-batches, create a |minibatchqueue| object.

% Mini-batch queues require a single datastore that outputs both the
% predictors and responses. Create array datastores containing the training
% BERT tokens and labels and combine them using the |combine| function.
dsXTrain = arrayDatastore(tokensTrain,"OutputType","same");
dsTTrain = arrayDatastore(TTrain);
cdsTrain = combine(dsXTrain,dsTTrain);

% Create a combined datastore for the validation data using the same steps.
dsXValidation = arrayDatastore(tokensValidation,"OutputType","same");
dsTValidation = arrayDatastore(TValidation);
cdsValidation = combine(dsXValidation,dsTValidation);

%%
% Create a mini-batch queue for the training data. Specify a mini-batch
% size of 32 and preprocess the mini-batches using the
% |preprocessPredictors| function, listed at the end of the example.
miniBatchSize = 32;
paddingValue = mdl.Tokenizer.PaddingCode;
maxSequenceLength = mdl.Parameters.Hyperparameters.NumContext;

mbqTrain = minibatchqueue(cdsTrain,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

%%%
% Create a mini-batch queue for the validation data using the same steps.
mbqValidation = minibatchqueue(cdsValidation,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

%%
% To speed up feature extraction. Convert the BERT model weights to
% gpuArray if a GPU is available.
if canUseGPU
    mdl.Parameters.Weights = dlupdate(@gpuArray,mdl.Parameters.Weights);
end

%%
% Convert the training sequences of BERT model tokens to a
% |N|-by-|embeddingDimension| array of feature vectors, where |N| is the
% number of training observations and |embeddingDimension| is the dimension
% of the BERT embedding.

featuresTrain = [];
reset(mbqTrain);
while hasdata(mbqTrain)
    X = next(mbqTrain);
    features = bertEmbed(X,mdl.Parameters);
    featuresTrain = [featuresTrain gather(extractdata(features))];
end

%%
% Transpose the training data to have size |N|-by-|embeddingDimension|.
featuresTrain = featuresTrain.';

%%
% Convert the validation data to feature vectors using the same steps.
featuresValidation = [];

reset(mbqValidation);
while hasdata(mbqValidation)
    X = next(mbqValidation);
    features = bertEmbed(X,mdl.Parameters);
    featuresValidation = cat(2,featuresValidation,gather(extractdata(features)));
end
featuresValidation = featuresValidation.';

%% Define Deep Learning Network
% Define a deep learning network that classifies the feature vectors.

numFeatures = mdl.Parameters.Hyperparameters.HiddenSize;
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Specify Training Options
% Specify the training options using the |trainingOptions| function.
% * Train with a mini-batch size of 64.
% * Shuffle the data every epoch.
% * Validate the network using the validation data.
% * Display the training progress in a plot and suppress the verbose
%   output.
opts = trainingOptions('adam',...
    "MiniBatchSize",64,...
    "ValidationData",{featuresValidation,dataValidation.Category},...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",0);

%% Train Network
% Train the network using the |trainNetwork| function.
net = trainNetwork(featuresTrain,dataTrain.Category,layers,opts);

%% Test Network
% Make predictions using the validation data and display the results in a
% confusion matrix.
YPredValidation = classify(net,featuresValidation);

figure
confusionchart(TValidation,YPredValidation)

%%
% Calculate the validation accuracy.
accuracy = mean(dataValidation.Category == YPredValidation)

%% Predict Using New Data
% Classify the event type of three new reports. Create a string array
% containing the new reports.
reportsNew = [ ...
    "Coolant is pooling underneath sorter."
    "Sorter blows fuses at start up."
    "There are some very loud rattling sounds coming from the assembler."];

%%
% Tokenize the text data using the same steps as the training documents.
tokensNew = encode(tokenizer,reportsNew);

%%
% Pad the sequences of tokens to the same length using the |padsequences| 
% function and pad using the tokenizer padding code.
XNew = padsequences(tokensNew,2,"PaddingValue",tokenizer.PaddingCode);

%%
% Classify the new sequences using the trained model.
featuresNew = bertEmbed(XNew,mdl.Parameters)';
featuresNew = gather(extractdata(featuresNew));
labelsNew = classify(net,featuresNew)

%% Supporting Functions

%%% Predictors Preprocessing Functions
% The |preprocessPredictors| function truncates the mini-batches to have
% the specified maximum sequence length, pads the sequences to have the
% same length. Use this preprocessing function to preprocess the predictors
% only.
function X = preprocessPredictors(X,paddingValue,maxSeqLen)

X = truncateSequences(X,maxSeqLen);
X = padsequences(X,2,"PaddingValue",paddingValue);

end

%%% BERT Embedding Function
% The |bertEmbed| function maps input data to embedding vectors and
% optionally applies dropout using the "DropoutProbability" name-value
% pair.
function Y = bertEmbed(X,parameters,args)

arguments
    X
    parameters
    args.DropoutProbability = 0
end

dropoutProbabilitiy = args.DropoutProbability;

Y = bert.model(X,parameters, ...
    "DropoutProb",dropoutProbabilitiy, ...
    "AttentionDropoutProb",dropoutProbabilitiy);

% To return single feature vectors, return the first element.
Y = Y(:,1,:);
Y = squeeze(Y);

end

