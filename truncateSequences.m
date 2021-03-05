function z = truncateSequences(z,maxSeqLen,nvp)
% truncateSequences   Truncates sequences with care not to remove special
% tokens. For use with BERT models.
% 
%   Z = truncateSequences(X,maxSeqLen) truncates an input array X to have
%   sequence length maxSeqLen. The input X is of size
%   1-by-SequenceLength-by-BatchSize. The truncation does not remove
%   separator tokens - entries of X equal to 103. If X is an encoded
%   sequence-pair then the first and second sequences are truncated to have
%   similar lengths.
% 
%   Z = truncateSequences(X,maxSeqLen,'SeparatorCode',sepCode) allows the
%   usage of a custom separator code for inputs X that we encoded using a
%   different technique to the default BERT tokenizer. The default value is
%   103. See also the SeparatorCode property of a BERTTokenizer.
%
% Example:
%   tokenizer = bert.tokenizer.BERTTokenizer();
%   sequences = tokenizer.encode(["Hello World!","I am a model."]);
%   truncatedSequences = truncateSequences(sequences,5)
%   % Note that truncatedSequences each still start with 102 and end with
%   103 - the encoded start and separator tokens.

% Copyright 2021 The MathWorks, Inc.
arguments
    z
    maxSeqLen (1,1) double {mustBeInteger,mustBeGreaterThan(maxSeqLen,2)}
    nvp.SeparatorCode (1,1) double {mustBeInteger} = 103
end
z = cellfun(@(z) iTruncateScalarSequence(z,maxSeqLen,nvp.SeparatorCode),z,'UniformOutput', false);
end

function z = iTruncateScalarSequence(z,maxSeqLen,sepCode)
if isa(z,'dlarray')
    z = extractdata(z);
end
idx = find(z==sepCode);
if idx == numel(z)
    z = iTruncateSingleSequence(z,maxSeqLen);
else
    z = iTruncateSequencePair(z,idx,maxSeqLen);
end
end

function z = iTruncateSingleSequence(z,maxSeqLen)
maxSeqLen = min(maxSeqLen-2,numel(z)-2);
indices = [1,2:(maxSeqLen+1),numel(z)];
z = z(indices);
end

function z = iTruncateSequencePair(z,idx,maxSeqLen)
z1 = z(1:idx);
z2 = z((idx+1):end);
n1 = numel(z1);
n2 = numel(z2);
N = n1+n2;
if N>maxSeqLen
    delta = N-maxSeqLen;
    if n1 > n2
        n1 = n1-delta;
        if n1 < n2
            gap = ceil((n2-n1)/2);
            n1 = n1+gap;
            n2 = n2-gap;
        end
    else
        n2 = n2-delta;
        if n2 < n1
            gap = ceil((n1-n2)/2);
            n1 = n1-gap;
            n2 = n2+gap;
        end
    end
    z1 = z1([1:(n1-1),numel(z1)]);
    z2 = z2([1:(n2-1),numel(z2)]);
end
z = [z1,z2];
end