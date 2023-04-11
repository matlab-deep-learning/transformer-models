classdef FullTokenizer < bert.tokenizer.internal.Tokenizer
    % FullTokenizer   A tokenizer based on word-piece tokenization.
    %
    %   tokenizer = FullTokenizer(vocabFile) constructs a FullTokenizer
    %   using the vocabulary specified in the newline delimited txt file
    %   vocabFile.
    %
    %   tokenizer = FullTokenizer(vocabFile,'PARAM1', VAL1, 'PARAM2', VAL2, ...)
    %   specifies the optional parameter name/value pairs:
    %
    %   'BasicTokenizer'       - Tokenizer used to split text into words.
    %                            If not specified, a default
    %                            BasicTokenizer is constructed.
    %
    %   'IgnoreCase'           - A logical value to control if the
    %                            FullTokenizer is case sensitive or not.
    %                            The default value is true.
    %
    %   FullTokenizer methods:
    %     tokenize - tokenize text
    %     encode   - encode tokens
    %     decode   - decode encoded tokens
    %
    % Example:
    %   % Save a file named fakeVocab.txt with the text on the next 3 lines:
    %   fake
    %   vo
    %   ##cab
    %
    %   % Now create a FullTokenizer
    %   tokenizer = bert.tokenizer.internal.FullTokenizer('fakeVocab.txt');
    %   tokens = tokenizer.tokenize("This tokenizer has a fake vocab")
    %   % Note that most tokens are unknown as they are not in the
    %   % vocabulary and neither are any sub-tokens. However "fake" is
    %   % detected and "vocab" is split into "vo" and "##cab".
    %   tokenizer.encode(tokens)
    %   % This returns the encoded form of the tokens - each token is
    %   % replaced by its corresponding line number in the fakeVocab.txt
    
    % Copyright 2021-2023 The MathWorks, Inc.
    
    properties(Access=private)
        Basic
        WordPiece
        Encoding
    end
    
    methods
        function this = FullTokenizer(vocab,nvp)
            % FullTokenizer   A tokenizer based on word-piece tokenization.
            %
            %   tokenizer = FullTokenizer(vocabFile) constructs a FullTokenizer
            %   using the vocabulary specified in the newline delimited txt file
            %   vocabFile.
            %
            %   tokenizer = FullTokenizer(vocabFile,'PARAM1', VAL1, 'PARAM2', VAL2, ...) specifies
            %   the optional parameter name/value pairs:
            %
            %   'BasicTokenizer'       - Tokenizer used to split text into words.
            %                            If not specified, a default
            %                            BasicTokenizer is constructed.
            %
            %   'IgnoreCase'           - A logical value to control if the
            %                            FullTokenizer is case sensitive or not.
            %                            The default value is true.
            %
            %   FullTokenizer methods:
            %     tokenize - tokenize text
            %     encode   - encode tokens
            %     decode   - decode encoded tokens
            %
            % Example:
            %   % Save a file named fakeVocab.txt with the text on the next 3 lines:
            %   fake
            %   vo
            %   ##cab
            %
            %   % Now create a FullTokenizer
            %   tokenizer = bert.tokenizer.internal.FullTokenizer('fakeVocab.txt');
            %   tokens = tokenizer.tokenize("This tokenizer has a fake vocab")
            %   % Note that most tokens are unknown as they are not in the
            %   % vocabulary and neither are any sub-tokens. However "fake" is
            %   % detected and "vocab" is split into "vo" and "##cab".
            %   tokenizer.encode(tokens)
            %   % This returns the encoded form of the tokens - each token is
            %   % replaced by its corresponding line number in the fakeVocab.txt
            arguments
                vocab
                nvp.BasicTokenizer = []
                nvp.IgnoreCase = true
            end
            if isempty(nvp.BasicTokenizer)
                % Default case
                this.Basic = bert.tokenizer.internal.BasicTokenizer('IgnoreCase',nvp.IgnoreCase);
            else
                mustBeA(nvp.BasicTokenizer,'bert.tokenizer.internal.Tokenizer');
                this.Basic = nvp.BasicTokenizer;
            end
            this.WordPiece = bert.tokenizer.internal.WordPieceTokenizer(vocab);
            this.Encoding = this.WordPiece.Vocab;
        end
        
        function toks = tokenize(this,txt)
            % tokenize   Tokenizes text.
            % 
            %   tokens = tokenize(tokenizer,text) tokenizes the input
            %   string text using the FullTokenizer specified by tokenizer.
            basicToks = this.Basic.tokenize(txt);
            toks = cell(numel(txt),1);
            for i = 1:numel(txt)
                theseBasicToks = textanalytics.unicode.UTF32(basicToks{i});
                theseSubToks = cell(numel(theseBasicToks),1);
                for j = 1:numel(theseBasicToks)
                    theseSubToks{j} = this.WordPiece.tokenize(theseBasicToks(j));
                end
                toks{i} = cat(2,theseSubToks{:});
            end
        end
        
        function idx = encode(this,tokens)
            % encode   Encodes tokens.
            %
            %   encoded = encode(tokenizer,tokens) encodes the string array
            %   tokens using the FullTokenizer specified by tokenizer.
            idx = this.Encoding.word2ind(tokens);
        end
        
        function tokens = decode(this,x)
            % decode   Decodes tokens.
            %
            %   decoded = decode(tokenizer,x) decodes the array of positive
            %   integers x into the string array decoded.
            tokens = this.Encoding.ind2word(x);
        end
    end
end
