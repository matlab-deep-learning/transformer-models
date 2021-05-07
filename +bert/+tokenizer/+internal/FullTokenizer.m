classdef FullTokenizer < bert.tokenizer.internal.Tokenizer
    % FullTokenizer   A tokenizer based on word-piece tokenization.
    %
    %   tokenizer = FullTokenizer(vocabFile) constructs a FullTokenizer
    %   using the vocabulary specified in the newline delimited txt file
    %   vocabFile.
    %
    %   tokenizer = FullTokenizer(vocabFile,'IgnoreCase',tf) controls if
    %   the FullTokenizer is case sensitive or not. The default value for
    %   tf is true.
    %
    %   FullTokenizer methods:
    %     tokenize - tokenize text
    %     encode   - encode tokens
    %     decode   - decode encoded tokens
    %
    % Example:
    %   % Save a file named vocab.txt with the text on the next 3 lines:
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
    
    % Copyright 2021 The MathWorks, Inc.
    
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
            %   tokenizer = FullTokenizer(vocabFile,'IgnoreCase',tf) controls if
            %   the FullTokenizer is case sensitive or not. The default value for
            %   tf is true.
            %
            %   FullTokenizer methods:
            %     tokenize - tokenize text
            %     encode   - encode tokens
            %     decode   - decode encoded tokens
            %
            % Example:
            %   % Save a file named vocab.txt with the text on the next 3 lines:
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
                nvp.IgnoreCase = true
            end
            this.Basic = bert.tokenizer.internal.BasicTokenizer('IgnoreCase',nvp.IgnoreCase);
            this.WordPiece = bert.tokenizer.internal.WordPieceTokenizer(vocab);
            this.Encoding = this.WordPiece.Vocab;
        end
        
        function toks = tokenize(this,txt)
            % tokenize   Tokenizes text.
            % 
            %   tokens = tokenize(tokenizer,text) tokenizes the input
            %   string text using the FullTokenizer specified by tokenizer.
            basicToks = this.Basic.tokenize(txt);
            basicToksUnicode = textanalytics.unicode.UTF32(basicToks);
            subToks = cell(numel(basicToks),1);
            for i = 1:numel(basicToks)
                subToks{i} = this.WordPiece.tokenize(basicToksUnicode(i));
            end
            toks = cat(2,subToks{:});
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