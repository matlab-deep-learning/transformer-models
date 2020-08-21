function sample = sampleFromCategorical(probabilities)
% sampleFromCategorical   Sample from a categorical distribution
%
%   sample = sampleFromCategorical(probabilities) returns an index sampled
%   from the categorical distribution represented by the input
%   probabilities.
%
%   Input:
%       probabilities   - A numClasses-by-1 vector of probabilities. The
%                         elements of this vector should sum to 1.
%
%   Output:
%       sample          - A number between 1 and numClasses that is sampled
%                         from the input probabilities.

cdf = cumsum(probabilities);
sample = find( cdf > rand );
sample = sample(1);

end