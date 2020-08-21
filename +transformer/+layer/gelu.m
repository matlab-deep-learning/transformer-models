function Z = gelu(X)
% gelu   GELU activation function
%
%   Z = gelu(X) computes the GELU activation function on X. The GELU
%   activation function is described in [1]. It is an element-wise
%   activation function, so the input and output are the same size.
%
%   References:
%
%   [1] Dan Hendrycks, Kevin Gimpel, "Gaussian Error Linear Units (GELUs)",
%       https://arxiv.org/abs/1606.08415

Z = 0.5*X.*( 1 + tanh( sqrt(2/pi)*(X+0.044715*(X.^3)) ) );
end