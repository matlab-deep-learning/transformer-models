function [x,untokenizedPieces,ismask] = encodeWithMaskToken(tok,str)
% encodeWithMaskToken   This function handles the case of encoding an input
% string that includes tokens such as [MASK].

% Copyright 2021 The MathWorks, Inc.
arguments
    tok bert.tokenizer.BERTTokenizer
    str (1,:) string
end
[seqs,untokenizedPieces] = arrayfun(@(s)encodeScalarString(tok,s),str,'UniformOutput',false);
x = padsequences(seqs,2,'PaddingValue',tok.PaddingCode);
maskCode = tok.MaskCode;
ismask = x==maskCode;
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