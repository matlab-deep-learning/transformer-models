classdef BERTTokenizer
    % BERTTokenizer   Construct a tokenizer to use with BERT
    % models.
    %
    %   tokenizer = BERTTokenizer()   Constructs a case-insensitive
    %   BERTTokenizer using the BERT-Base vocabulary file.
    %
    %   tokenizer = BERTTokenizer(vocabFile)   Constructs a
    %   case-insensitive BERTTokenizer using the file vocabFile as
    %   the vocabulary.
    %
    %   tokenizer = BERTTokenizer(vocabFile,'IgnoreCase',tf)
    %   Constructs a BERTTokenizer which is case-sensitive or not
    %   according to the scalar logical tf. The default is true.
    %
    %   BERTTokenizer properties:
    %     FullTokenizer  - The underlying word-piece tokenizer.
    %     PaddingToken   - The string "[PAD]"
    %     StartToken     - The string "[CLS]"
    %     SeparatorToken - The string "[SEP]"
    %     MaskToken      - The string "[MASK]"
    %     PaddingCode    - The encoded PaddingToken
    %     StartCode      - The encoded StartToken
    %     SeparatorCode  - The encoded SeparatorToken
    %     MaskCode       - The encoded MaskToken
    %
    %   BERTTokenizer methods:
    %     tokenize     - Tokenize strings
    %     encode       - Tokenize and encode strings
    %     encodeTokens - Encode pre-tokenized token sequences
    %     decode       - Decode an encoded sequence to string
    %
    % Example:
    %   tokenizer = bert.tokenizer.BERTTokenizer();
    %   sequences = tokenizer.encode("Hello World!")
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(Constant)
        PaddingToken = "[PAD]"
        StartToken = "[CLS]"
        SeparatorToken = "[SEP]"
        MaskToken = "[MASK]"
    end
    
    properties(GetAccess=public,SetAccess=private)
        FullTokenizer
        PaddingCode
        SeparatorCode
        StartCode
        MaskCode
    end
    
    methods
        function this = BERTTokenizer(vocabFile,nvp)
            % BERTTokenizer   Construct a tokenizer to use with BERT
            % models.
            %
            %   tokenizer = BERTTokenizer()   Constructs a case-insensitive
            %   BERTTokenizer using the BERT-Base vocabulary file.
            %
            %   tokenizer = BERTTokenizer(vocabFile)   Constructs a
            %   case-insensitive BERTTokenizer using the file vocabFile as
            %   the vocabulary.
            %
            %   tokenizer = BERTTokenizer(vocabFile,'IgnoreCase',tf)
            %   Constructs a BERTTokenizer which is case-sensitive or not
            %   according to the scalar logical tf. The default is true.
            %
            %   BERTTokenizer properties:
            %     FullTokenizer  - The underlying word-piece tokenizer.
            %     PaddingToken   - The string "[PAD]"
            %     StartToken     - The string "[CLS]"
            %     SeparatorToken - The string "[SEP]"
            %     MaskToken      - The string "[MASK]"
            %     PaddingCode    - The encoded PaddingToken
            %     StartCode      - The encoded StartToken
            %     SeparatorCode  - The encoded SeparatorToken
            %     MaskCode       - The encoded MaskToken
            %
            %   BERTTokenizer methods:
            %     tokenize     - Tokenize strings
            %     encode       - Tokenize and encode strings
            %     encodeTokens - Encode pre-tokenized token sequences
            %     decode       - Decode an encoded sequence to string
            %
            % Example:
            %   tokenizer = bert.tokenizer.BERTTokenizer();
            %   sequences = tokenizer.encode("Hello World!")
            arguments
                vocabFile (1,1) string {mustBeFile} = bert.internal.getSupportFilePath("base","vocab.txt")
                nvp.IgnoreCase (1,1) logical = true
            end
            ignoreCase = nvp.IgnoreCase;
            this.FullTokenizer = bert.tokenizer.internal.FullTokenizer(vocabFile,'IgnoreCase',ignoreCase);
            this.PaddingCode = this.FullTokenizer.encode(this.PaddingToken);
            this.SeparatorCode = this.FullTokenizer.encode(this.SeparatorToken);
            this.StartCode = this.FullTokenizer.encode(this.StartToken);
            this.MaskCode = this.FullTokenizer.encode(this.MaskToken);
        end
        
        function tokens = tokenize(this,text_a,text_b)
            % tokenize   Tokenizes a batch of strings and adds special
            % tokens for the BERT model.
            %
            %   tokens = tokenize(bertTokenizer,text) tokenizes text using
            %   the BERTTokenizer specified by bertTokenizer. The input
            %   text is a string array. The output tokens is a cell-array
            %   where tokens{i} is a string array corresponding to the
            %   tokenized text(i). The start token and separator token are
            %   preprended and appended respectively.
            %
            %   tokens = tokenize(bertTokenizer,text_a,text_b) tokenizes
            %   the sentence-pairs (text_a,text_b). Here tokens{i}
            %   corresponds to the tokenized form of the sentence pair
            %   (text_a(i),text_b(i)), including the separator token
            %   between the tokenized text_a(i) and text_b(i). The inputs
            %   text_a and text_b must have the same number of elements.
            %
            % Example:
            %   tokenizer = bert.tokenizer.BERTTokenizer;
            %   tokens = tokenizer.tokenize("Hello world!")
            arguments
                this
                text_a string
                text_b string = string.empty()
            end
            if ~isempty(text_b) && numel(text_a)~=numel(text_b)
                error("bert:tokenizer:SentencePairNumelMismatch","For sentence-pairs, both inputs must have the same number of elements");
            end
            inputShape = size(text_a);
            text_a = reshape(text_a,[],1);
            text_b = reshape(text_b,[],1);
            tokenize = @(text) this.FullTokenizer.tokenize(text);
            tokens = arrayfun(tokenize,text_a,'UniformOutput',false);
            if ~isempty(text_b)
                tokens_b = arrayfun(tokenize,text_b,'UniformOutput',false);
                tokens = cellfun(@(tokens_a,tokens_b) [tokens_a,this.SeparatorToken,tokens_b], tokens, tokens_b, 'UniformOutput', false);
            end
            tokens = cellfun(@(tokens) [this.StartToken, tokens, this.SeparatorToken], tokens, 'UniformOutput', false);
            tokens = reshape(tokens,inputShape);
        end
        
        function x = encodeTokens(this,toks)
            % encodeTokens   Encodes pre-tokenized tokens.
            %
            %   seqs = encodeTokens(bertTokenizer,tokens) encodes the
            %   cell-array tokens into a cell-array of sequences of
            %   positive integers seqs. For a BERTTokenizer specified by
            %   bertTokenizer the output of
            %   encodeTokens(bertTokenizer,tokenize(bertTokenizer,text)) is
            %   equivalent to encode(bertTokenizer,text).
            %
            % Example:
            %   tokenizer = bert.tokenizer.BERTTokenizer;
            %   tokens = tokenizer.tokenize("Hello world!");
            %   sequences = tokenizer.encodeTokens(tokens)
            x = cellfun(@(tokens) this.FullTokenizer.encode(tokens), toks, 'UniformOutput', false);
        end
        
        function x = encode(this,text_a,text_b)
            % encode   Tokenizes and encodes strings.
            %
            %   x = encode(bertTokenizer,text) will tokenize
            %   and encode the string array specified by text into a
            %   cell-array of sequences of positive integers. The encoded
            %   start token and separator token are prepended and appended
            %   to each sequence respectively.
            %
            %   x = encode(tok,text_a,text_b) For a sentence-pair task
            %   each input sentence text_a and text_b is encoded, then the
            %   encoded sequences are joined on an encoded separator token.
            %   The inputs text_a and text_b must have the same number of
            %   elements.
            %
            % Example:
            %   tokenizer = bert.tokenizer.BERTTokenizer;
            %   sequences = tokenizer.encode(["Hello world!"; ...
            %   "I am a model."])
            arguments
                this
                text_a string
                text_b string = string.empty()
            end
            if ~isempty(text_b) && numel(text_a)~=numel(text_b)
                error("bert:tokenizer:SentencePairNumelMismatch","For sentence-pairs, both inputs must have the same number of elements");
            end
            tokens = this.tokenize(text_a,text_b);
            x = this.encodeTokens(tokens);
        end
        
        function text = decode(this,x)
            % decode   Decode sequences of encoded tokens back to their
            % string form and join on a space.
            %
            %   text = decode(bertTokenizer,x) decodes an input of encoded
            %   tokens x into string tokens and joins these on a space
            %   character to create the output text. The input x can be a
            %   cell-array of sequences of positive integers or an array
            %   of positive integers with size
            %   1-by-numInputSubWords-by-numObs. Note that decode is not
            %   the inverse to encode as the output of decode joins tokens
            %   on a space character, and the tokenization performed by
            %   encode is not inverted by the decode method.
            %
            % Example:
            %   tokenizer = bert.tokenizer.BERTTokenizer();
            %   sequences = tokenizer.encode("Hello World!");
            %   decoded = tokenizer.decode(sequences)
            
            if ~iscell(x)
                % assume x is CTB
                x = mat2cell(x,size(x,1),size(x,2),ones(size(x,3),1));
                % return as a column vector.
                x = reshape(x,[],1);
            end
            tokens = cellfun(@(s) this.FullTokenizer.decode(s), x, 'UniformOutput', false);
            text = cellfun(@(x) join(x," "), tokens);
        end
    end
end