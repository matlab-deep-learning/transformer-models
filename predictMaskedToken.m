function out = predictMaskedToken(mdl,str)
% predictMaskedToken   Given a BERT language model, predict the most likely
% tokens for masked tokens.
%
%   out = predictMaskedToken(mdl, text) returns the string out which
%   replaces instances of mdl.Tokenizer.MaskToken in the string text with
%   the most likely token according to the BERT model mdl.

% Copyright 2021 The MathWorks, Inc.
arguments
    mdl {mustBeA(mdl,'struct')}
    str {mustBeText}
end
str = string(str);
inSize = size(str);
str = str(1:end);
[seqs,pieces] = arrayfun(@(s)encodeScalarString(mdl.Tokenizer,s),str,'UniformOutput',false);
x = padsequences(seqs,2,'PaddingValue',mdl.Tokenizer.FullTokenizer.encode(mdl.Tokenizer.PaddingToken));
maskCode = mdl.Tokenizer.FullTokenizer.encode(mdl.Tokenizer.MaskToken);
ismask = x==maskCode;
x = dlarray(x);
probs = bert.languageModel(x,mdl.Parameters);
maskedProbs = extractdata(probs(:,ismask));
[~,sampleIdx] = max(maskedProbs,[],1);
predictedTokens = mdl.Tokenizer.FullTokenizer.decode(sampleIdx);
out = strings(numel(seqs),1);
numMaskPerSeq = sum(ismask,2);
maskStartIdx = 1;
for i = 1:numel(seqs)
    startIdx = maskStartIdx;
    maskStartIdx = maskStartIdx+numMaskPerSeq(i);
    out(i) = rebuildScalarString(pieces{i},predictedTokens(startIdx:(startIdx+numMaskPerSeq(i)-1)));
end
out = reshape(out,inSize);
end

function [x,pieces] = encodeScalarString(tok,str)
pieces = split(str,tok.MaskToken);
fulltok = tok.FullTokenizer;
maskCode = fulltok.encode(tok.MaskToken);
x = [];

for i = 1:numel(pieces)
    tokens = fulltok.tokenize(pieces(i));
    if ~isempty(tokens)
        % "" tokenizes to empty - awkward
        x = cat(2,x,fulltok.encode(tokens));
    end
    if i<numel(pieces)
        x = cat(2,x,maskCode);
    end
end
x = [fulltok.encode(tok.StartToken),x,fulltok.encode(tok.SeparatorToken)];
end

function out = rebuildScalarString(pieces,predictedTokens)
out = "";
for i = 1:(numel(pieces)-1)
    out = strcat(out,pieces(i),predictedTokens(i));
end
out = strcat(out,pieces(end));
end