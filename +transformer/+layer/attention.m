function [A, present] = attention(X, past, weights, hyperParameters, nvp)
% attention   Full Multi-head Attention
%
%   [A, present] = attention(X, past, weights, hyperParameters) computes a
%   multi-head attention block on X as outlined in Section 3.2.2 and Figure
%   2 in [1]. See below for details of inputs and outputs.
%
%   Inputs:
%       X               - A (numFeatures*numHeads)-by-numInputSubwords-by-numObs
%                         input array.
%       past            - A numFeatures-by-numPastSubwords-by-numHeads-by-numObs-by-2
%                         array. This contains the 'keys' and 'values' for
%                         past subwords. These are needed to predict future
%                         outputs in an autoregressive manner. 'keys' are
%                         stored in past(:,:,:,:,1) and 'values' are stored
%                         in past(:,:,:,:,2).
%       weights         - The weights for the full multi-head attention
%                         block stored in a struct. This includes:
%                           - attn_c_attn_w_0: A weight matrix for the
%                             first fully connected layer.
%                           - attn_c_attn_b_0: A bias vector for the first
%                             fully connected layer.
%                           - attn_c_proj_w_0: A weight matrix for the
%                             final fully connected layer.
%                           - attn_c_proj_b_0: A bias vector for the final
%                             fully connected layer.
%       numHeads        - The number of attention heads. This is a
%                         hyper-parameter.
%
%   Outputs:
%       A               - A (numFeatures*numHeads)-by-numInputSubwords-by-numObs
%                         output array.
%       present         - A numFeatures-by-numAllSubwords-by-numHeads-by-numObs-by-2
%                         array. This contains the 'keys' and 'values' that
%                         are created from inputs. These need to passed
%                         back in as the 'past' input if we want to predict
%                         future outputs in an autoregressive manner. 'keys'
%                         are stored in present(:,:,:,:,1) and 'values' are
%                         stored in present(:,:,:,:,2).
%
%   [A, present] = attention(X, past, weights, hyperParameters, 'PARAM1',
%   VAL1, 'PARAM2', VAL2, ...) specifies the optional parameter name/value
%   pairs:
%
%     'CausalMask'  - A scalar logical to turn causal masking on or off. Causal
%                     masking prevents tokens at time T attending to tokens
%                     at time S<T. The default is true.
%
%     'Dropout'     - The dropout probability for the attention
%                     probabilities. The default is 0.
%
%     'InputMask'   - A logical mask to mask attending to particular
%                     tokens, for example padding tokens. The default is
%                     [], interpreted as not applying any masking.
%
%   References:
%
%   [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
%       Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention
%       Is All You Need", https://arxiv.org/abs/1706.03762
arguments
    X
    past
    weights
    hyperParameters
    nvp.CausalMask (1,1) logical = true
    nvp.Dropout (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(nvp.Dropout,1)} = 0
    nvp.InputMask = []
end

% Use a fully connected layer to generate queries, keys and values from the
% input.
C = transformer.layer.convolution1d( X, ...
    weights.attn_c_attn_w_0, ...
    weights.attn_c_attn_b_0 );

% Split the results into Q (Query), K (Keys) and V (Values).
splitSize = size(C,1)/3;
Q = C(1:splitSize,:,:);
K = C((splitSize+1):(2*splitSize),:,:);
V = C((2*splitSize+1):(3*splitSize),:,:);

% Split heads
Q = iSplitHeads(Q, splitSize, hyperParameters.NumHeads);
K = iSplitHeads(K, splitSize, hyperParameters.NumHeads);
V = iSplitHeads(V, splitSize, hyperParameters.NumHeads);

% Use the past
if ~isempty(past)
    PK = past(:,:,:,:,1);
    PV = past(:,:,:,:,2);
    K = cat(2,PK,K);
    V = cat(2,PV,V);
end

% Set present. Note that this is done differently from the original
% implementation which sets the value of present before the previous if
% statement
present = cat(5,K,V);

A = transformer.layer.multiheadAttention(Q,K,V,'CausalMask',nvp.CausalMask,'Dropout',nvp.Dropout,'InputMask',nvp.InputMask);

A = iMergeHeads(A);

A = transformer.layer.convolution1d( A, ...
    weights.attn_c_proj_w_0, ...
    weights.attn_c_proj_b_0 );
end

function Z = iSplitHeads(X, splitSize, numHeads)
% We permute the data to put the dimension for the heads last, so that we
% can use batched matrix multiplication to compute attention for all of the
% heads at once.
%
% X     - A (numFeatures*numHeads)-by-numSubwords-by-numObs array.
% Z     - A numFeatures-by-numSubwords-by-numHeads-by-numObs array.
X = reshape(X, splitSize/numHeads, numHeads, [], size(X,3));
Z = permute(X,[1 3 2 4]);
end

function Z = iMergeHeads(X)
% X     - A numFeatures-by-numSubwords-by-numHeads-by-numObs array.
% Z     - A (numFeatures*numHeads)-by-numSubwords-by-numObs array.
X = permute(X, [1 3 2 4]);
Z = reshape(X, size(X,1)*size(X,2), [], size(X,4));
end