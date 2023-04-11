classdef(SharedTestFixtures = {
        DownloadBERTFixture}) tFullTokenizer < matlab.mock.TestCase
    % tFullTokenizer   Unit tests for the FullTokenizer.
    
    % Copyright 2021-2023 The MathWorks, Inc.
    
    methods(Test)
        function matchesExpectedTokenization(test)
            % Test the tokenizer
            vocabFile = bert.internal.getSupportFilePath("base","vocab.txt");
            tok = bert.tokenizer.internal.FullTokenizer(vocabFile);
            
            % Create a string to tokenize.
            str = "UNwant"+compose("\x00E9")+"d,running.";
            exp_toks = {["unwanted",",","running","."]};
            act_toks = tok.tokenize(str);
            test.verifyEqual(act_toks,exp_toks);
        end

        function errorsIfBasicTokenizerIsNotTokenizer(test)
            vocabFile = bert.internal.getSupportFilePath("base","vocab.txt");
            makeTok = @() bert.tokenizer.internal.FullTokenizer(vocabFile,Basic=vocabFile);
            test.verifyError(makeTok,"MATLAB:validators:mustBeA");
        end

        function canSetBasicTokenizer(test)
            [mock,behaviour] = test.createMock(?bert.tokenizer.internal.Tokenizer);
            test.assignOutputsWhen(withAnyInputs(behaviour.tokenize),"hello");
            vocabFile = bert.internal.getSupportFilePath("base","vocab.txt");
            tok = bert.tokenizer.internal.FullTokenizer(vocabFile,BasicTokenizer=mock);
            toks = tok.tokenize("anything");
            test.verifyEqual(toks,{"hello"}); %#ok<STRSCALR> 
        end
    end
end