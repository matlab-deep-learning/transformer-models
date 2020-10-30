function [logits, presents] = model(X, pasts, parameters)
% model   A GPT-2 model
%
%   [logits, presents] = model(X, pasts, parameters) performs prediction
%   with a GPT-2 model on the input X. X is a
%   1-by-numInputSubwords-by-numObs array of tokenized text, and the model
%   returns an array logits that is 50257-by-numInputSubwords-by-numObs.
%   This array can be used to predict the next subword. See below for more
%   details of inputs and outputs.
%
%   Inputs:
%       X                 - A 1-by-numInputSubwords-by-numObs array. This
%                           array is a tokenized sentence. It should be
%                           created using the tokenizer for GPT-2.
%       pasts             - A numLayers-by-1 cell array containing "keys"
%                           and "values" for the attention layers. These
%                           come from the previous subwords in the text we
%                           are processing. If there are no previous words,
%                           this should be an empty cell.
%       parameters        - The parameters for the GPT-2 model in a struct.
%                           It has two fields, 'Hyperparameters' and
%                           'Weights'. 'Hyperparameters' has the following
%                           fields:
%                             - NumHeads: The number of attention heads in
%                               each multi-head attention layer.
%                             - NumLayers: The number of blocks in the
%                               GPT-2 model.
%                             - NumContext: The size of the context
%                               embedding.
%                           'Weights' includes the folloeing fields:
%                             - wte_0: A numFeatures-by-50257 embedding
%                               matrix for subwords.
%                             - wpe_0: A numFeatures-by-numContext
%                               positional embedding matrix for subwords.
%                             - h<X>: Structs containing weights for the
%                               transformer blocks where <X> is the number
%                               for the block. Numbers start from 0.
%                             - ln_f_g_0: Weight vector for final layer
%                               normalization.
%                             - ln_f_b_0: Bias vector for final layer
%                               normalization.
%
%   Outputs:
%       logits            - A 50257-by-numInputSubwords-by-numObs array of
%                           logits (pre-softmax outputs). If we apply
%                           softmax to this array, we get the probabilities
%                           for the next subword. However, we usually want
%                           to do more pre-processing before doing this
%                           (like taking the top-K entries). 50257 is the
%                           number of subwords in the vocabulary for
%                           GPT-2's tokenizer.
%       presents          - A numLayers-by-1 cell array containing "keys"
%                           and "values" from the attention blocks. We feed
%                           these back in as the 'pasts' input.

hyperparameters     = parameters.Hyperparameters;
weights             = parameters.Weights;

% Apply the embedding. If there are inputs for the "past", we need to
% offset the position embedding to account for this.
% Word embedding
seqLen = size(X, 2);
h = weights.wte_0(:, X);
h = reshape(h, size(h,1), seqLen, []);
% Positional embedding
positionOffset = size(pasts{1},2);
h = h + weights.wpe_0(:, positionOffset + (1:seqLen) );

% Run the layers
presents = cell(hyperparameters.NumLayers,1);
for i = 1:hyperparameters.NumLayers
    layerName = ['h' num2str(i-1)];
    [h, present] = gpt2.layer.block( h, pasts{i}, weights.(layerName), ...
        hyperparameters );
    presents{i} = present;
end

h = transformer.layer.normalization( h, ...
    weights.ln_f_g_0, ...
    weights.ln_f_b_0 );

% Calculate logits (50257-by-numInputSubwords-by-numObs)
logits = dlmtimes(weights.wte_0', h);

end