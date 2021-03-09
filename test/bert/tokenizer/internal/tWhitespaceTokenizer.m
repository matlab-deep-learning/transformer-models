classdef tWhitespaceTokenizer < matlab.unittest.TestCase
    % tWhitespaceTokenizer   Unit tests for WhitespaceTokenizer.
    
    % Copyright 2021 The MathWorks, Inc.
    
    methods(Test)
        function canTokenize(test)
            tok = bert.tokenizer.internal.WhitespaceTokenizer();
            str = "foo bar baz ";
            exp_out = ["foo","bar","baz"];
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);            
        end       
    end
end
