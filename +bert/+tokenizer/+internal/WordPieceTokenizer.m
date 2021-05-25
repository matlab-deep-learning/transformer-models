classdef WordPieceTokenizer < bert.tokenizer.internal.Tokenizer
    % WordPieceTokenizer   Constructs a Word Piece Tokenizer from a given
    %   vocab.txt file.
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess=private)
        Vocab
        Unk
        MaxChar
    end
    
    properties(Constant,Access=private)
        WhitespaceTokenizer = bert.tokenizer.internal.WhitespaceTokenizer()
    end
    
    methods
        function this = WordPieceTokenizer(vocab,nvp)
            % tok = WordPieceTokenizer(vocab)   Constructs a
            % WordPieceTokenizer with vocabulary file vocab.
            %
            % Name-Value Pair Arguments:
            %   'UnknownToken'   - String to use for unknown tokens.
            %                      Default is "[UNK]".
            %
            %   'MaxTokenLength' - Maximum length of a token. Tokens longer
            %                      than this are replaced by
            %                      'UnknownToken'.
            %                      Default is 200.
            arguments
                vocab {mustBeFileOrEncoding}
                nvp.UnknownToken (1,1) string = "[UNK]"
                nvp.MaxTokenLength (1,1) {mustBePositive,mustBeInteger} = 200
            end
            this.Unk = nvp.UnknownToken;
            this.MaxChar = nvp.MaxTokenLength;
            this.Vocab = this.parseVocab(vocab);
        end
        
        function tokens = tokenize(this,utext)
            arguments
                this
                utext
            end
            tokens = string.empty();
            sub = textanalytics.unicode.UTF32();
            for i = 1:numel(utext)
                token = utext(i);
                if numel(token.Data)>this.MaxChar
                    tokens = [tokens,this.Unk]; %#ok
                    continue
                end
                isBad = false;
                start = 1;
                subTokens = string.empty();
                while start<(numel(token.Data)+1)
                    finish = numel(token.Data);
                    currentSub = [];
                    while start<finish+1                        
                        sub.Data = token.Data(start:finish);
                        if start>1
                            sub.Data = [uint32('##'),sub.Data];
                        end
                        strForm = sub.string();
                        if this.Vocab.isVocabularyWord(strForm)
                            currentSub = strForm;
                            break
                        end
                        finish = finish-1;
                    end
                    if isempty(currentSub)
                        isBad = true;
                        break
                    end
                    subTokens(end+1) = currentSub;%#ok
                    start = finish+1;
                end
                
                if isBad
                    tokens = [tokens, this.Unk];%#ok
                else
                    tokens = [tokens, subTokens];%#ok
                end
            end
        end
    end
    
    methods(Access=private)
        function vocab = parseVocab(~,vocab)
            if isa(vocab,'wordEncoding')
                return
            end
            if exist(vocab,'file')~=2
                error("Unknown vocabulary file");
            end
            fid = fopen(vocab,'r','n','utf-8');
            c = fread(fid,Inf);
            fclose(fid);
            c = native2unicode(c,'utf-8');%#ok
            words = splitlines(c').';
            empties = cellfun(@isempty,words);
            words(empties) = [];
            vocab = wordEncoding(words);
        end
    end
end

function mustBeFileOrEncoding(x)
if ~isa(x,'wordEncoding')
    mustBeFile(x);
end
end