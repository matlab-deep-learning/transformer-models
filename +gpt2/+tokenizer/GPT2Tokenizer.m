classdef GPT2Tokenizer < handle
    % GPT2Tokenizer   Object for encoding text so it can be fed to GPT2
    
    properties(SetAccess = private)
        % Encoding
        Encoding
        
        % BPERanks
        BPERanks
        
        % Cache
        Cache = containers.Map()
    end
    
    properties(Constant)
        % TokenizationExpression   Regular expression used for tokenization
        %
        %   This is the regular expression used for the first stage of
        %   tokenization. It was hard-coded by the creators of GPT-2. It
        %   appears to apply a tokenization rule that can be summarised as
        %   follows:
        %
        %   A token is one of the following things:
        %
        %   - An exact string match for 's, 't, 're, 've, 'm, 'll, or 'd.
        %     This means common contractions in words like don't and you'll
        %     will get split into their own tokens.
        %   - Zero or one spaces followed by one or more Unicode letters.
        %   - Zero or one spaces followed by one or more Unicode numbers.
        %   - Zero or one spaces followed by one or more things that are
        %     not whitespace, a Unicode letter or a Unicode number.
        %   - One or more whitespace characters not followed by a
        %     non-whitepace character. This is tricky to understand, but
        %     basically it means that a string with a word preceeded by
        %     several spaces like '   Hello' will get split into '  ' and
        %     ' Hello'.
        %   - One or more whitespace characters.
        %
        %   Note that we have had to modify the original expression, which
        %   is shown below:
        %
        %       '''s|''t|''re|''ve|''m|''ll|''d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'
        %
        %   MATLAB's regexp function does not support the \p flag, so we
        %   have replaced it with something with equivalent functionality.
        TokenizationExpression = '''s|''t|''re|''ve|''m|''ll|''d| ?((?![\d_])\w)+| ?\d+| ?(_|[^\s\w])+|\s+(?!\S)|\s+';
        
        % ByteEncoder   Encodes bytes into a set of 256 Unicode characters
        %
        %   The size of the output vocabulary from this encoder determines
        %   the size of the embedding needed by the GPT-2 transformer
        %   model. The creators of GPT-2 wanted to keep this at around
        %   50,000. However, they wanted to be able to encode any Unicode
        %   string. Unicode has potentially hundreds of thousands of
        %   characters. So to keep the overall vocabulary low, we go
        %   through an extra encoding stage:
        %   
        %   - The raw Unicode string (which can contain any Unicode
        %     character) is converted into UTF-8 bytes. Note that UTF-8 is
        %     a variable length encoding scheme, so each character can get
        %     mapped to between 1 to 4 bytes.
        %   - These individual bytes are then mapped to a restricted
        %     vocabulary of 256 Unicode characters. ByteEncoder defines
        %     this mapping.
        ByteEncoder = iBytesToUnicode()
    end
    
    methods
        function this = GPT2Tokenizer(~, ~)
            % Read in the vocabulary. The UTF-8 part is really important to
            % make this work on Windows.
            vocabFile = gpt2.internal.getSupportFilePath("gpt2_vocab.bpe");
            fid = fopen(vocabFile, 'r', 'n', 'UTF-8');
            bpeData = textscan(fid,'%s', 'Delimiter', '\n');
            fclose(fid);
            
            bpeData = bpeData{1};   % textscan always reads everything in a cell
            bpeData(1) = [];        % Delete the first line we read in (it's a comment)
            
            % Split the bpe data into two columns.
            this.BPERanks = split(string(bpeData));
            
            % Read in the encoding data. The UTF-8 part is really important
            % to make this work on Windows.
            encoderFile = gpt2.internal.getSupportFilePath("gpt2_encoder.txt");
            fid = fopen(encoderFile, 'r', 'n', 'UTF-8');
            encoderData = textscan(fid,'%s', 'Delimiter', '\n');
            fclose(fid);
            
            encoderData = encoderData{1};
            
            % Set the encoding
            this.Encoding = string(encoderData);
        end
        
        function numericTokens = encode(this, text)
            
            % Note that this function returns tokens with indices that
            % begin at 1. The Python implementation indexes from 0.
            
            % Step 1: Apply regular expression to split text into words.
            % See the comment for 'TokenizationExpression' for more detail
            % on what is going on here.
            [inputTokens, ~] = regexp( ...
                text, ...
                this.TokenizationExpression, ...
                'match', 'split');
            
            % Step 2: The incoming text is Unicode. Unicode has a huge set
            % of characters. We do not want our BPE algorithm to deal with
            % a huge set of characters, because that will inflate the BPE
            % vocabulary. So we need to reduce the set of characters. We do
            % this by converting the Unicode text to the UTF-8 encoding,
            % and then we replace each UTF-8 byte with another Unicode
            % character, out of a set of 256 Unicode characters. This will
            % mean that our original Unicode string which could have
            % contained any Unicode character will now contain only one of
            % 256 characters.
            encodedTokens = cellfun( @(x)unicode2native(x, 'UTF-8'), ...
                inputTokens, 'UniformOutput', false );
            encodedTokens = cellfun( @(x)this.ByteEncoder(x+1), ...
                encodedTokens, 'UniformOutput', false );
            
            % Step 3: Do the BPE encoding on a per word basis. Words are
            % either left as they are, or for rare words we split them into
            % word fragments.
            bpeTokens = cellfun(@(x)this.bpe(x), encodedTokens, 'UniformOutput', false);
            
            % Step 4: Look up each word or word fragment and replace it
            % with a number.
            numericTokens = [];
            for i = 1:numel(bpeTokens)
                bpeTokensSplit = split(bpeTokens{i});
                for j = 1:numel(bpeTokensSplit)
                    numericTokens = [numericTokens find(this.Encoding == bpeTokensSplit(j))]; %#ok<AGROW>
                end
            end
        end
        
        function text = decode(this, numericTokens)
            
            % Note that this function expects tokens that begin at 1!
            
            % Step 1: Turn tokens into text
            text = join(this.Encoding(numericTokens),'');
            
            % Step 2: Replace characters with byte values
            [~,text] = max( char(text) == this.ByteEncoder' );
            text = text -1;
            
            % Step 3: Decode byte values as UTF-8
            text = native2unicode(text, 'UTF-8');
        end
    end
    
    methods(Access = private)
        function word = bpe(this, token)
            if this.Cache.isKey(token)
                word = this.Cache(token);
            elseif isempty(token)
                word = token;
            else
                wordFragments = string(num2cell(token));
                pairs = iGetPairs(wordFragments);
                
                while true
                    matches = [];
                    for i = 1:numel(pairs)
                        match = find(sum(pairs{i} == this.BPERanks, 2) == 2);
                        matches = [matches match]; %#ok<AGROW>
                    end
                    minIndex = min(matches);
                    if isempty(minIndex)
                        break;
                    end
                    bigram = this.BPERanks(minIndex,:);
                    
                    first = bigram(1);
                    second = bigram(2);
                    newWordFragments = [];
                    i = 1;
                    while i < length(wordFragments)+1
                        j = find( ...
                            wordFragments == first & ...
                            [zeros(1,(i-1)) ones(1,length(wordFragments)-i+1)]);
                        if isempty(j)
                            newWordFragments = [newWordFragments wordFragments(i:end)]; %#ok<AGROW>
                            break
                        else
                            newWordFragments = [newWordFragments wordFragments(i:(j(1)-1))]; %#ok<AGROW>
                            i = j(1);
                        end
                        
                        if wordFragments(i) == first && ...
                                i < length(wordFragments) && ...
                                wordFragments(i+1) == second
                            newWordFragments = [newWordFragments first+second]; %#ok<AGROW>
                            i = i + 2;
                        else
                            newWordFragments = [newWordFragments wordFragments(i)]; %#ok<AGROW>
                            i = i + 1;
                        end
                    end
                    
                    % We have a new word because we have merged some of the
                    % word fragments. If there is only one element in
                    % 'wordFragments', we have merges all of the fragments,
                    % and can stop now, so we break. Otherwise, we generate
                    % pairs again, and start the process again.
                    wordFragments = newWordFragments;
                    if numel(wordFragments) == 1
                        break;
                    else
                        pairs = iGetPairs(wordFragments);
                    end
                end
                
                word = join(wordFragments, ' ');
                this.Cache(token) = word;
            end
        end
    end
end

function cs = iBytesToUnicode()
% Note that the third character here is not the letter i! It is the
% extended Unicode character corresponding to the number 161.
%cs = ['!':'~' '¡':'¬' '®':'ÿ'];
cs = char([33:126 161:172 174:255]);
bs = double(cs);
n = 0;
for b = 0:255
    if ~any(b == bs)
        bs = [bs b]; %#ok<AGROW>
        cs = [cs 256+n]; %#ok<AGROW>
        n = n + 1;
    end
end
[~,sortedIndices] = sort(bs);
cs = cs(sortedIndices);
end

function pairs = iGetPairs(wordFragments)
numLetters = length(wordFragments);
pairIndices = [1:(numLetters-1); 2:numLetters]';
pairIndices = mat2cell(pairIndices, ones(numLetters-1,1), 2);
pairs = cellfun(@(x)wordFragments(x), pairIndices, ...
    'UniformOutput', false);
pairs = cellfun(@(x)[string(x(1)) string(x(2))], pairs, ...
    'UniformOutput', false);
end