function A = multiheadAttention(Q, K, V,nvp)
% multiheadAttention   Multi-head Attention
%
%   A = multiheadAttention(Q, K, V) computes scaled dot product attention
%   for multiple attention heads as outlined in [1] (see Section 3.2.1 and
%   Figure 2). Note that this function computes the attention for multiple
%   attention heads at once for efficiency. Q is a collection of query
%   matrices, K is a collection of key matrices and V is a collection of
%   value matrices. The output A is a collection of attention matrices. See
%   below for details.
%
%   Inputs:
%       Q   - numFeatures-by-numInputSubWords-by-numHeads-by-numObs array of queries.
%       K   - numFeatures-by-numAllSubWords-by-numHeads-by-numObs array of keys.
%       V   - numFeatures-by-numAllSubWords-by-numHeads-by-numObs array of values.
%
%   Outputs:
%       A   - numFeatures-by-numInputSubWords-by-numHeads-by-numObs array of attention matrices.
%
%   A = multiheadAttention(Q, K, V, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies the optional parameter name/value pairs:
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
    Q
    K
    V
    nvp.CausalMask (1,1) logical = true
    nvp.Dropout (1,1) double {mustBeNonnegative,mustBeLessThanOrEqual(nvp.Dropout,1)} = 0
    nvp.InputMask = []
end

% We compute attention weights by taking the product between Q and K
% matrices. W is numAllSubWords-by-numInputSubWords-by-numHeads. Each
% element of W is the dot product of a query vector from Q and a key vector
% from K.
W = dlmtimes(permute(K, [2 1 3 4]), Q);

% Divide by square root of d
W = W./sqrt(size(Q,1));

% Apply masking
W = transformer.layer.maskAttentionWeights(W,'CausalMask',nvp.CausalMask,'InputMask',nvp.InputMask);

% Apply softmax
W = softmax(W, 'DataFormat', 'CTUB');

% Apply dropout
W = transformer.layer.dropout(W,nvp.Dropout);

% We compute the attention by taking products between the attention weights
% W and V. A is numFeatures-by-numInputSubWords-by-numHeads. One
% interpretation of A is that it is the expected value of V according to
% the probability distribution given by W.
A = dlmtimes(V, W);
end