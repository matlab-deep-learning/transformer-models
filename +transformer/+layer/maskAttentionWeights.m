function W = maskAttentionWeights(W)
% maskAttentionWeights   Function for masking attention weights
%
%   W = maskAttentionWeights(W) applies masking to a set of attention
%   matrices W (prior to the application of the softmax function). This
%   prevents subwords from attending to other subwords that come AFTER
%   them, which is necessary when we are using scaled dot product attention
%   for a 'decoder' transformer that predicts future outputs.
%
%   Inputs:
%       W   - A numAllSubwords-by-numInputSubwords-by-numHeads array.
%
%   Outputs:
%       W   - A numAllSubwords-by-numInputSubwords-by-numHeads array.

numAllSubwords = size(W,1);
numInputSubwords = size(W,2);
numPreviousSubwords = numInputSubwords-numAllSubwords;

% The mask is numAllSubwords-by-numInputSubwords. Input subwords should not
% attend to other input subwords that come after them. The matrix will have
% zeroes in positions we want to mask out, and ones in positions we want to
% keep.
mask = triu( ...
    ones([numAllSubwords numInputSubwords],'like', extractdata(W)), ...
    numPreviousSubwords );

% We want to make sure that when softmax is applied to the matrix W, the
% probability for the things we want to mask out is zero. We do this by
% setting those entries to a large negative value.
W = W.*mask - (1e10)*(1-mask);

end