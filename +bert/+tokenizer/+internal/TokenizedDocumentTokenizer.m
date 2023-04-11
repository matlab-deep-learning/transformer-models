classdef TokenizedDocumentTokenizer < bert.tokenizer.internal.Tokenizer
    % TokenizedDocumentTokenizer   Implements a word-level tokenizer using
    % tokenizedDocument. 

     % Copyright 2023 The MathWorks, Inc.
    
    properties
        TokenizedDocumentOptions
        IgnoreCase
    end
    
    methods
        function this = TokenizedDocumentTokenizer(varargin,args)
            arguments(Repeating)
                varargin
            end
            arguments
                args.IgnoreCase (1,1) logical = true
            end
            this.IgnoreCase = args.IgnoreCase;
            this.TokenizedDocumentOptions = varargin;
        end
        
        function toks = tokenize(this,txt)
            arguments
                this
                txt (1,:) string
            end
            if this.IgnoreCase
                txt = lower(txt);
            end
            t = tokenizedDocument(txt,this.TokenizedDocumentOptions{:});
            toks = doc2cell(t);
        end
    end
end