function  z = embedding(x,types,positions,w,dropout)
% embedding   The BERT embeddings of encoded tokens, token types and token
% positions.
%
%   Z = embedding(X,types,positions,weights,dropoutProbability) computes 
%   the embedding of encoded tokens X, token types specified by types, and 
%   token positions. Inputs X, types and positions are 
%   1-by-numInputTokens-by-numObs unformatted dlarray-s. The types take
%   values 1 or 2. The weights input is a struct of embedding weights such
%   as mdl.Parameters.Weights.embeddings where mdl = bert(). The
%   dropoutProbability is a scalar double between 0 and 1 corresponding to
%   the post-embedding dropout probability.

% Copyright 2021 The MathWorks, Inc.
wordEmbedding = embed(x,w.word_embeddings,'DataFormat','CTB');
typeEmbedding = embed(types,w.token_type_embeddings,'DataFormat','CTB');
positionEmbedding = embed(positions,w.position_embeddings,'DataFormat','CTB');
z = wordEmbedding+typeEmbedding+positionEmbedding;
z = transformer.layer.normalization(z,w.LayerNorm.gamma,w.LayerNorm.beta);
z = transformer.layer.dropout(z,dropout);
end