function z = languageModelHead(z,p,word_embeddings)
% languageModelHead   The standard BERT operations for masked
% language modeling.
%
%   Z = languageModelHead(X,languageModelWeights,wordEmbeddingMatrix)
%   applies the language model operations to an input X. The input X must
%   be an unformatted dlarray of size
%   hiddenSize-by-numInputTokens-by-numObs. The languageModelWeights must
%   be a struct with fields 'transform' and 'LayerNorm' such as the
%   mdl.Parameters.Weights.masked_LM struct where mdl = bert(). The
%   wordEmbeddingMatrix must be the word embedding matrix used by the bert
%   model such as mdl.Parameters.Weights.embeddings.word_embeddings where
%   mdl = bert().

% Copyright 2021 The MathWorks, Inc.
z = transformer.layer.convolution1d(z,p.transform.kernel,p.transform.bias);
z = transformer.layer.gelu(z);
z = transformer.layer.normalization(z,p.LayerNorm.gamma,p.LayerNorm.beta);
z = dlmtimes(word_embeddings.',z);
z = z + p.output.bias;
z = softmax(z,'DataFormat','CTB');
end