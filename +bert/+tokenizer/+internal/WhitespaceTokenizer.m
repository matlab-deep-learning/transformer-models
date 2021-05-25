classdef WhitespaceTokenizer < bert.tokenizer.internal.Tokenizer
    % WhitespaceTokenizer   The simplest type of tokenization, split on
    % whitespace characters.
    
    % Copyright 2020 The MathWorks, Inc.
    
    methods
        function text = tokenize(~,text)
            % tokens = tokenize(tok,str)   Returns an array of tokens formed
            %                              by splitting str on whitespace.
            arguments
                ~
                text
            end
            text = strip(text);
            text = split(text).';
        end
    end
end
