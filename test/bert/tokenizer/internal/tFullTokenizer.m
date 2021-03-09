classdef(SharedTestFixtures = {
        DownloadBERTFixture}) tFullTokenizer < matlab.unittest.TestCase
    % tFullTokenizer   Unit tests for the FullTokenizer.
    
    % Copyright 2021 The MathWorks, Inc.
    
    methods(Test)
        function matchesExpectedTokenization(test)
            % Test the tokenizer
            vocabFile = bert.internal.getSupportFilePath("base","vocab.txt");
            tok = bert.tokenizer.internal.FullTokenizer(vocabFile);
            
            % Create a string to tokenize.
            str = "UNwant"+compose("\x00E9")+"d,running.";
            exp_toks = ["unwanted",",","running","."];
            act_toks = tok.tokenize(str);
            test.verifyEqual(act_toks,exp_toks);
        end
    end
end