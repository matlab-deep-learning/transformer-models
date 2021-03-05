function [toks,probs] = predictMaskedToken(mdl,x,maskIdx,k)
arguments
    mdl
    x
    maskIdx
    k (1,1) double {mustBePositive,mustBeInteger} = 1
end
probs = bert.languageModel(x,mdl.Parameters);
probs = extractdata(probs(:,maskIdx));
[~,idx] = maxk(probs,k);
toks = mdl.Tokenizer.FullTokenizer.decode(idx);
end
    
    