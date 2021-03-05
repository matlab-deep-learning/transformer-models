%% Classify Text Data Using BERT
% This example shows how to classify text data using a deep learning BERT model.
% 
% It is based on <https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html 
% https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html> 
% 
% 
%% Import Data
% Import the factory reports data. This data contains labeled textual descriptions 
% of factory events. To import the text data as strings, specify the text type 
% to be |'string'|.

filename = "factoryReports.csv";
data = readtable(filename,'TextType','string');
head(data)
%% 
% The goal of this example is to classify events by the label in the |Category| 
% column. To divide the data into classes, convert these labels to categorical.

data.Category = categorical(data.Category);
%% 
% View the distribution of the classes in the data using a histogram.

figure
histogram(data.Category);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")
%% 
% The next step is to partition it into sets for training and validation. Partition 
% the data into a training partition and a held-out partition for validation and 
% testing. Specify the holdout percentage to be 20%.

mdl = bert;
tok = mdl.Tokenizer;
data.Tokens = tok.encode(data.Description);

cvp = cvpartition(data.Category,'Holdout',0.2);
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);
%% 
% Extract the text data and labels from the partitioned tables.

textDataTrain = dataTrain.Description;
textDataValidation = dataValidation.Description;
YTrain = dataTrain.Category;
YValidation = dataValidation.Category;
tokensTrain = dataTrain.Tokens;
tokensValidation = dataValidation.Tokens;

%% 
% To check that you have imported the data correctly, visualize the training 
% text data using a word cloud.

figure
wordcloud(textDataTrain);
title("Training Data")
%% View BERT Token Codes
% View the first few tokenized training documents.

tokensTrain{1:5}
%% Plot Document Lengths
% Length is the number of BERT tokens.

documentLengths = cellfun(@length,tokensTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")
%% 
% Most of the training documents have fewer than 15 tokens. 

%% Prepare Training Data
% Prepare the data for training in minibatchqueue-s.

cats = categories(YTrain);
numClasses = numel(cats);
if canUseGPU
    mdl.Parameters.Weights = dlupdate(@gpuArray,mdl.Parameters.Weights);
end

dsX = arrayDatastore(tokensTrain,"OutputType","same");
dsXVal = arrayDatastore(tokensValidation,"OutputType","same");
dsY = arrayDatastore(YTrain);
dsYVal = arrayDatastore(YValidation);
cds = combine(dsX,dsY);
cdsVal = combine(dsXVal,dsYVal);

paddingValue = mdl.Tokenizer.PaddingCode;
maxSeqLen = mdl.Parameters.Hyperparameters.NumContext;
minibatchFunction = @(x,y) minibatchFcn(x,y,paddingValue,maxSeqLen);
minibatchSize = 32;

mbq = minibatchqueue(cds,2,...
  "MiniBatchFcn",minibatchFunction,...
  "MiniBatchSize",minibatchSize);

mbqVal = minibatchqueue(cdsVal,2,...
  "MiniBatchFcn",minibatchFunction,...
  "MiniBatchSize",minibatchSize);

%% Fine Tune BERT
% To get the most out of the BERT model you need to fine-tune the BERT
% weights for your task. 

% First we initialize some weights for the classifier to be applied after
% the BERT embedding.
outputSize = mdl.Parameters.Hyperparameters.HiddenSize;
mdl.Parameters.Weights.classifier.kernel = dlarray(randn(numClasses, outputSize));
mdl.Parameters.Weights.classifier.bias = dlarray(zeros(numClasses, 1));

numEpochs = 4;
params = mdl.Parameters;
vel = [];
avgG = [];
avgSqg = [];
iter = 0;
lr = 1e-5;
figure;
h = animatedline;
hval = animatedline('Color','r');
drawnow;
xlabel("Iteration");
ylabel("Loss");

for epoch = 1:numEpochs
  mbq.shuffle();
  while mbq.hasdata
    iter = iter+1;
    [x,y] = mbq.next();

    % Compute loss and gradients. Update model parameters.
    [L,dL] = dlfeval(@modelGradients,x,y,params);
    [params.Weights, avgG,avgSqg] = adamupdate(params.Weights,dL,avgG,avgSqg,iter,lr);
    
    % Plot loss.
    if mod(iter-1,3)==0
      addpoints(h,iter,double(gather(extractdata(L))));
      drawnow;
    end
    
    % Compute and plot validation loss.
    if mod(iter-1,10)==1
      valLoss = 0;
      valIter = 0;
      mbqVal.shuffle();
      while mbqVal.hasdata
        valIter = valIter+1;
        [xVal,yVal] = mbqVal.next();
        LVal = dlfeval(@modelGradients,xVal,yVal,params);
        valLoss = valLoss + gather(extractdata(LVal));
      end
      mbqVal.reset();
      addpoints(hval,iter,double(valLoss/valIter));
      drawnow;
    end
  end
  mbq.reset();
  mbq.shuffle();
end

%% Evaluate the fine-tuned model
% Evaluate the fine tuned model on the validation data to compare it to
% simply training a classifier on the BERT embeddings.
dropout = 0;
numCorrect = 0;
while mbqVal.hasdata
    [x,y] = mbqVal.next();
    z = model(x,params,dropout);
    [~,prediction] = max(z);
    [~,target] = max(y);
    numCorrect = numCorrect + sum(prediction==target,'all');
end
validationAccuracy = numCorrect/size(dataValidation,1)

%% Predict Using New Data
% Classify the event type of three new reports. Create a string array containing 
% the new reports.

reportsNew = [ ...
    "Coolant is pooling underneath sorter."
    "Sorter blows fuses at start up."
    "There are some very loud rattling sounds coming from the assembler."];
%% 
% Tokenize the text data using the same steps as the training documents.

documentsNew = tok.encode(reportsNew);
%% 
% Convert the text data to sequences using |doc2sequence| with the same options 
% as when creating the training sequences.

XNew = padsequences(documentsNew,2,'PaddingValue',1);
%% 
% Classify the new sequences using the trained model.
dropout = 0;
z = model(XNew,params,dropout);
[~,YNew] = max(z);
YNew = gather(extractdata(YNew));
labelsNew = categorical(YNew, 1:length(cats), cats)'
%% Supporting Functions
% 

function z = poolSequence(z)
% A simple pooling strategy - use the first sequence token. This works with
% BERT due to the self-attention mechanism, each token can attend to each
% other token. 
% We do not use bert.layer.pooler here as the pretrained pooling weights 
% for BERT are not trained for this classification task.
z = squeeze(z(:,1,:));
end

function z = bertEmbed(x,params,dropout)
z = bert.model(x,params,'DropoutProb',dropout,'AttentionDropoutProb',dropout);
z = poolSequence(z);
end

function z = model(x,params,dropout)
z = bertEmbed(x,params,dropout);
% z is a HiddenSize x BatchSize array.
% We can apply any operation that reduces this to NumClasses x BatchSize
% bert.layer.classifier is simply a fullyconnect
z = bert.layer.classifier(z,params.Weights.classifier);
end

function [L,dL] = modelGradients(x,y,params)
z = model(x,params,0.1);
z = softmax(z,'DataFormat','CB');
L = crossentropy(z,y,'DataFormat','CB');
dL = dlgradient(L,params.Weights);
end

function [x,y] = minibatchFcn(x,y,paddingValue,maxSeqLen)
x = truncateSequences(x,maxSeqLen);
x = padsequences(x,2,'PaddingValue',paddingValue);
y = cat(2,y{:});
y = onehotencode(y,1);
end

%% 
% _Copyright 2021 The MathWorks, Inc._