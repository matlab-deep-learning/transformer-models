function [clf,score] = sentimentModel(x,p)
% sentimentModel   The FinBERT sentiment analysis model.
%
% clf = finbert.sentimentModel(x,p)   Given an input x of size
%   1-by-numInputTokens-by-numObs and FinBERT parameters struct p, the output
%   clf is a categorical of size 1-by-numObs with categories
%   "positive","neutral" or "negative".
%
% [clf,score] = finbert.sentimentModel(x,p)   The second output is a sentiment
%   score in the range [-1,1].

% Copyright 2021 The MathWorks, Inc.
if ~isfield(p.Weights,'classifier')
    error("finbert:sentimentAnalysis:NoClassifier","Parameters do not include classifier weights");
end

z = bert.model(x,p);
y = bert.layer.classifierHead(z,p.Weights.pooler,p.Weights.classifier);
logits = softmax(y,'DataFormat','CB');
[~,clf_i] = max(extractdata(logits));
classes = ["positive","negative","neutral"];
score = logits(1,:) - logits(2,:);
clf = categorical(classes(clf_i),classes);
end