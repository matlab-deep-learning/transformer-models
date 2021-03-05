function z = block(z,weights,hyperParameters,nvp)
% block   Transformer block for BERT
% 
%   Z = block(X,weights,hyperParameters) computes the BERT style
%   transformer block on the input X as described in [1]. Here X is a
%   (numFeatures*numHeads)-by-numInputSubwords array. The weights and
%   hyperParameters must be structs in the same format as returned by the
%   bert() function.
%
%   Z = block(X,weights,hyperParameters,'PARAM1',VAL1,'PARAM2',VAL2)
%   specifies the optional parameter name/value pairs:
%
%     'HiddenDropout'    - The dropout probability to be applied between
%                          the self attention mechanism and the residual
%                          connection. The default is 0.
%
%     'AttentionDropout' - The dropout probability to be applied to the
%                          attention probabilities. The default is 0.
%
%     'InputMask'        - A logical mask to be used in the attention
%                          mechanism, for example to block attending to
%                          padding tokens. The default is [], no masking is
%                          applied.
%  
% References:
% [1] https://arxiv.org/abs/1810.04805

% Copyright 2021 The MathWorks, Inc.
arguments
  z
  weights
  hyperParameters
  nvp.HiddenDropout (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(nvp.HiddenDropout,1)} =  0
  nvp.AttentionDropout (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(nvp.AttentionDropout,1)} = 0
  nvp.InputMask = []
end  
z = attention(z,weights.attention,hyperParameters.NumHeads,nvp.AttentionDropout,nvp.HiddenDropout,nvp.InputMask);
z = ffn(z,weights.feedforward,nvp.HiddenDropout);
end

function z = attention(z,w,num_heads,attentionDropout,dropout,mask)
% The self attention part of the transformer layer.
layer_input = z;

% Get weights
Q_w = w.query.kernel;
Q_b = w.query.bias;
K_w = w.key.kernel;
K_b = w.key.bias;
V_w = w.value.kernel;
V_b = w.value.bias;

% Put weights into format for transformer.layer.attention
weights.attn_c_attn_w_0 = cat(1,Q_w,K_w,V_w);
weights.attn_c_attn_b_0 = cat(1,Q_b,K_b,V_b);
weights.attn_c_proj_w_0 = w.output.kernel;
weights.attn_c_proj_b_0 = w.output.bias;
hyperparameters.NumHeads = num_heads;
z = transformer.layer.attention(z,[],weights,hyperparameters,'CausalMask',false,'Dropout',attentionDropout,'InputMask',mask);

% Dropout
z = transformer.layer.dropout(z,dropout);

% Residual connection
z = layer_input+z;

% Layer normalize.
z = transformer.layer.normalization(z,w.LayerNorm.gamma,w.LayerNorm.beta);
end

function z = ffn(z,w,dropout)
% The feed-forward network part of the transformer layer.

% Weights for embedding in higher dimensional space
int_w = w.intermediate.kernel;
int_b = w.intermediate.bias;

% Weights for projecting back down to original space
out_w = w.output.kernel;
out_b = w.output.bias;

% Create weights struct for multiLayerPerceptron
weights.mlp_c_fc_w_0 = int_w;
weights.mlp_c_fc_b_0 = int_b;
weights.mlp_c_proj_w_0 = out_w;
weights.mlp_c_proj_b_0 = out_b;
ffn_out = transformer.layer.multiLayerPerceptron(z,weights);

% Dropout
ffn_out = transformer.layer.dropout(ffn_out,dropout);

% Layer normalize.
out_g = w.LayerNorm.gamma;
out_b = w.LayerNorm.beta;
z = transformer.layer.normalization(ffn_out+z,out_g,out_b);
end