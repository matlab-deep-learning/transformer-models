function varargout = model(x,parameters,nvp)
% model   A FinBERT model forward pass.
%
%   Z = model(X,parameters) performs inference with a BERT model on the
%   input X. X is a 1-by-numInputTokens-by-numObs array of encoded tokens.
%   The return is an array Z of size
%   (NumHeads*HeadSize)-by-numInputTokens-by-numObs. Each
%   Z(:,i,j) corresponds to the BERT embedding of input token X(1,i,j).
%
%   Z = model(X,parameters,'PARAM1', VAL1, 'PARAM2', VAL2, ...) specifies
%   the optional parameter name/value pairs:
%
%      'InputMask'             - A logical mask with the same size as X.
%                                The mask should be false at indices 
%                                (i,j,k) for which X(i,j,k) corresponds to
%                                padding, and true elsewhere. The default
%                                is empty, for which the padding is
%                                inferred by the entries of X that match
%                                the PaddingCode name-value pair.
%
%      'DropoutProb'           - The probability of dropout for the output
%                                activation. It is standard to set this to a
%                                non-zero value during training, for example
%                                0.1. The default is 0.
%
%      'AttentionDropoutProb'  - The probability of dropout used in the
%                                attention layer. It is standard to set
%                                this to a non-zero value during training,
%                                for example 0.1. The default is 0.
%
%      'Outputs'               - Specify the indices of the layers to
%                                return outputs from as a vector of
%                                positive integers, or 'last' to specify
%                                the final encoder layer only. The default
%                                is 'last'.
%
%      'SeparatorCode'         - The positive integer corresponding to the
%                                separator token. The default is 103, as
%                                specified in the default BERT vocab.txt.
%
%      'PaddingCode'           - The positive integer corresponding to the
%                                padding token. The default is 1, as
%                                specified in the default BERT vocab.txt.

% Copyright 2021 The MathWorks, Inc.
arguments
    x dlarray {mustBeNumericDlarray,mustBeNonempty}
    parameters {mustBeA(parameters,'struct')}
    nvp.InputMask {mustBeALogicalOrDlarrayLogical} = logical.empty()
    nvp.DropoutProb (1,1) {mustBeNonnegative,mustBeLessThanOrEqual(nvp.DropoutProb,1),mustBeNumeric} = 0
    nvp.AttentionDropoutProb (1,1) {mustBeNonnegative,mustBeLessThanOrEqual(nvp.AttentionDropoutProb,1),mustBeNumeric} = 0
    nvp.Outputs {mustBePositive,mustBeLessThanOrEqualNumLayers(nvp.Outputs,parameters),mustBeInteger,mustBeNumeric} = parameters.Hyperparameters.NumLayers
    nvp.PaddingCode (1,1) {mustBePositive,mustBeInteger,mustBeNumeric} = 1
    nvp.SeparatorCode (1,1) {mustBePositive,mustBeInteger,mustBeNumeric} = 103
end

% Pure wrapper.
varargout = cell(max(nargout,1),1);
[varargout{:}] = bert.model(x,parameters,...
    'InputMask',nvp.InputMask,...
    'DropoutProb',nvp.DropoutProb,...
    'AttentionDropoutProb',nvp.AttentionDropoutProb,...
    'Outputs',nvp.Outputs,...
    'PaddingCode',nvp.PaddingCode,...
    'SeparatorCode',nvp.SeparatorCode);
end

function mustBeLessThanOrEqualNumLayers(x,params)
mustBeLessThanOrEqual(x,params.Hyperparameters.NumLayers);
end

function mustBeALogicalOrDlarrayLogical(val)
if isa(val,'dlarray')
    val = extractdata(val);
end
mustBeA(val,'logical');
end

function mustBeNumericDlarray(val)
mustBeA(val,'dlarray');
mustBeNumeric(extractdata(val));
end