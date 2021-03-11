function z = dropout(z,p)
% z = dropout(x,p)  Applies inverted dropout with probability p to input x.
arguments
    z
    p (1,1) double {mustBeNonnegative, mustBeLessThan(p,1)}
end

% Copyright 2020 The MathWorks, Inc.

ps = rand(size(z),'like',z);
dropoutMask = (1-(ps<p)) ./ (1-p);
z = z.*dropoutMask;
end