%% Fine-Tune Pretrained BERT Model
% This example shows how to fine-tune a pretrained BERT model for text
% classification.
%
% To get the most out of a pretrained BERT model, you can retrain and
% fine-tune the BERT parameters weights for your task.
%
% This example shows how to fine-tune a pretrained BERT model to classify
% failure events given a data set of factory reports.

%% Load Pretrained BERT Model
% Load a pretrained BERT model using the |bert| function. The model
% consists of a tokenizer that encodes text as sequences of integers, and a
% structure of parameters.
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
% Extract the training text data, labels, and encoded BERT tokens from the
% partitioned tables.
textDataTrain = dataTrain.Description;
TTrain = dataTrain.Category;
tokensTrain = dataTrain.Tokens;

%%
% To check that you have imported the data correctly, visualize the
% training text data using a word cloud.

figure
wordcloud(textDataTrain);
title("Training Data")

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

%% Initialize Model Parameters
% Initialize the weights for the classifier to apply after the BERT
% embedding.
outputSize = mdl.Parameters.Hyperparameters.HiddenSize;
mdl.Parameters.Weights.classifier.kernel = dlarray(randn(numClasses, outputSize));
mdl.Parameters.Weights.classifier.bias = dlarray(zeros(numClasses, 1));

%% Specify Training Options
% Train for 4 epochs with a mini-batch size of 32. Train with a learning
% rate of 0.00001.
numEpochs = 4;
miniBatchSize = 32;
learnRate = 1e-5;

%% Train Model
% Fine tune the model parameters using a custom training loop.

%%
% Create a mini-batch queue for the training data. Preprocess the
% mini-batches using the |preprocessMiniBatch| function, listed at the end
% of the example and discard any partial mini-batches.
paddingValue = mdl.Tokenizer.PaddingCode;
maxSequenceLength = mdl.Parameters.Hyperparameters.NumContext;

mbqTrain = minibatchqueue(cdsTrain,2,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X,Y) preprocessMiniBatch(X,Y,paddingValue,maxSequenceLength), ...
    "PartialMiniBatch","discard");

%%
% Initialize training progress plot.
figure
C = colororder;
lineLossTrain = animatedline("Color",C(2,:));

ylim([0 inf]);
xlabel("Iteration");
ylabel("Loss");

%%
% Initialize parameters for the Adam optimizer.
trailingAvg = [];
trailingAvgSq = [];

%% 
% Extract the model parameters from the pretrained BERT model.
parameters = mdl.Parameters;

%% 
% Train the model using a custom training loop.
%
% For each epoch, shuffle the mini-batch queue and loop over mini-batches
% of data. At the end of each iteration, update the training progress plot.
%
% For each iteration:
% * Read a mini-batch of data from the mini-batch queue. 
% * Evaluate the model gradients and loss using the |dlfeval| and
%   |modelGradients| functions.  
% * Update the network parameters using the |adamupdate| function.
% * Update the training plot.

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbqTrain);
    
    % Loop over mini-batches
    while hasdata(mbqTrain)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbqTrain);
        
        % Evaluate loss and gradients.
        [loss,gradients] = dlfeval(@modelGradients,X,T,parameters);
        
        % Update model parameters.
        [parameters.Weights,trailingAvg,trailingAvgSq] = adamupdate(parameters.Weights,gradients, ...
            trailingAvg,trailingAvgSq,iteration,learnRate);
        
        % Update training plot.
        loss = double(gather(extractdata(loss)));
        addpoints(lineLossTrain,iteration,loss);
        
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

%% Test Network
% Test the network using the held-out validation data.

%%
% Extract the encoded tokens and labels from the validation data table.
tokensValidation = dataValidation.Tokens;
TValidation = dataValidation.Category;

%% 
% Create an array datastore containing the encoded tokens.
dsXValidation = arrayDatastore(tokensValidation,"OutputType","same");

%%
% Create a mini-batch queue for the validation data. Preprocess the
% mini-batches using the |preprocessPredictors| function, listed at the end
% of the example.
mbqValidation = minibatchqueue(dsXValidation,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

%% 
% Make predictions using the |modelPredictions| function, listed at the end
% of the example, and display the results in a confusion matrix.
YPredValidation = modelPredictions(parameters,mbqValidation,classes);

figure
confusionchart(TValidation,YPredValidation)

%% Predict Using New Data
% Classify the event type of three new reports.

%%
% Create a string array containing the new reports.
reportsNew = [ ...
    "Coolant is pooling underneath sorter."
    "Sorter blows fuses at start up."
    "There are some very loud rattling sounds coming from the assembler."];

%%
% Encode the text data as sequences of tokens using the BERT model
% tokenizer.
tokensNew = encode(tokenizer, reportsNew);

%%
% Create a mini-batch queue for the new data. Preprocess the mini-batches
% using the |preprocessPredictors| function, listed at the end of the
% example.
dsXNew = arrayDatastore(tokensNew,"OutputType","same");

mbqNew = minibatchqueue(dsXNew,1,...
    "MiniBatchSize",miniBatchSize, ...
    "MiniBatchFcn",@(X) preprocessPredictors(X,paddingValue,maxSequenceLength));

%%
% Make predictions using the |modelPredictions| function, listed at the end
% of the example.
YPredNew = modelPredictions(parameters,mbqNew,classes)

%% Supporting Functions

%%% Mini-batch Preprocessing Function.
% The |preprocessMiniBatch| function preprocess the predictors using the
% |preprocessPredictors| function and then encodes the labels as encoded
% vectors. Use this preprocessing function to preprocess both predictors
% and labels.
function [X,T] = preprocessMiniBatch(X,T,paddingValue,maxSequenceLength)

X = preprocessPredictors(X,paddingValue,maxSequenceLength);
T = cat(2,T{:});
T = onehotencode(T,1);

end

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

%%% Model Function
% The function |model| performs a forward pass of the classification model.
function Y = model(X,parameters,dropout)

Y = bertEmbed(X,parameters,"DropoutProbability",dropout);

weights = parameters.Weights.classifier.kernel;
bias = parameters.Weights.classifier.bias;
Y = fullyconnect(Y,weights,bias,"DataFormat","CB");

end

%%% Model Gradients Function
% The |modelGradients| function performs a forward pass of the
% classification model and returns the model loss and gradients of the loss
% with respect to the learnable parameters.
function [loss,gradients] = modelGradients(X,T,parameters)

dropout = 0.1;
Y = model(X,parameters,dropout);
Y = softmax(Y,"DataFormat","CB");
loss = crossentropy(Y,T,"DataFormat","CB");
gradients = dlgradient(loss,parameters.Weights);

end

%%% Model Predictions Function
% The |modelPredictions| function makes predictions by iterating over
% mini-batches of data.
function predictions = modelPredictions(parameters,mbq,classes)

predictions = [];

dropout = 0;

reset(mbq);

while hasdata(mbq)
    
    dlX = next(mbq);
    dlYPred = model(dlX,parameters,dropout);
    dlYPred = softmax(dlYPred,"DataFormat","CB");
    
    YPred = onehotdecode(dlYPred,classes,1)';
    
    predictions = [predictions; YPred];
end

end



