function types = inferTypeID(x,separatorCode)
% infer the typeIDs from a CTB unlabeled array x
xsz = size(x);
types = ones(xsz);
sepId = x==separatorCode;
if isa(sepId,'dlarray')
    sepId = extractdata(sepId);
end
% Find which observations have >1 separator - when there is 1 separator,
% any padding is considered "type 1".
cs = cumsum(sepId,2);
obsNeedsType2 = cs(:,end,:)>1;
% Type 2 tokens are those between the first (exclusive) and second
% separator (inclusive) if a second separator was present.
type2positions = circshift(cs==1,1) & obsNeedsType2;
types(type2positions) = 2;
end