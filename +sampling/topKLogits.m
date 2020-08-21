function logits = topKLogits(logits, topK)
% topKLogits   Return the top K logits
%
%   logits = topKLogits(logits, k) will return a vector of logits where any
%   classes that are not in the top K largest values will be supressed.
%   Values are supressed by setting them to large negative values.
%
%   Inputs:
%       logits  - A numClasses-by-1 vector of logits.
%       k       - The number of values to 'keep'. Everything outside of the
%                 top K values will be supressed. Note for many typical use
%                 cases this parameter can have a big effect.
%
%   Outputs:
%       logits  - A numClasses-by-1 vector of logits.

if isa(logits, 'dlarray')
    extractedLogits = extractdata(logits);
else
    extractedLogits = logits;
end
[~,classRanks] = sort(extractedLogits, 1, 'descend');

notTopKIndices = ( (topK+1):size(classRanks,1) )';
notTopKRows = classRanks(notTopKIndices,:);
notTopKColumns = repmat(1:size(extractedLogits,2),size(notTopKIndices,1),1);

notTopKIndices = sub2ind(size(extractedLogits), notTopKRows, notTopKColumns);

% We want to make sure that when softmax is applied to the logits, the
% probability for the classes that are not in the top K are zero. We do
% this by setting entries that are not in the top K to large negative
% values.
logits( notTopKIndices ) = -1e10;

end