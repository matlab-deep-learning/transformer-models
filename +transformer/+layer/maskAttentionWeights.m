function W = maskAttentionWeights(W,nvp)
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
%
%   W = maskAttentionWeights(W, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies the optional parameter name/value pairs:
%
%      'CausalMask' - A scalar logical to turn causal masking on or off. 
%                     Causal masking prevents tokens at time T attending to
%                     tokens at time S<T. The default is true.
% 
%      'InputMask'  - A logical mask to mask attending to particular
%                     tokens, for example padding tokens. The default is
%                     [], interpreted as not applying any masking.

% Copyright 2020-2021 The MathWorks, Inc.
arguments
    W
    nvp.CausalMask (1,1) logical = true
    nvp.InputMask = []
end

if nvp.CausalMask
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

if ~isempty(nvp.InputMask)
    % mask is in CTB format, permute to attention shape - TT(head)B
    mask = permute(nvp.InputMask,[2,1,4,3]);
    % expand mask to W size - T and B dimensions should be good, just
    % repeat over extra T and heads dimension.
    mask = repmat(mask,[1,size(W,2),size(W,3),1]);
    W = W-1e4.*(~mask);
end
end