classdef(Abstract) Tokenizer
    %Tokenizer  Interface for tokenizer classes
    methods(Abstract)
        tokenize(this,text)
    end
end