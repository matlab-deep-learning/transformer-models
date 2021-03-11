classdef(SharedTestFixtures={
        DownloadFinBERTFixture}) ...
        tsentimentModel < matlab.unittest.TestCase
    % tsentimentAnalysis   Unit test for finbert.sentiment_analysis
    
    % Copyright 2021 The MathWorks, Inc.
    
    methods(Test)        
        function canUseSentimentAnalysisModel(test)
            % Verify we can use finbert.sentiment_analysis with the
            % finbert() model.
            
            mdl = finbert();
            strs = ["Good day on the markets.";"My portfolio is down by 10%"];
            seqs = mdl.Tokenizer.encode(strs);
            x = padsequences(seqs,2,'PaddingValue',mdl.Tokenizer.PaddingCode);
            [sentClass,sentScore] = finbert.sentimentModel(x,mdl.Parameters);
            % Regression test against some current values.
            test.verifyEqual(sentClass,categorical(["neutral","negative"]));
            test.verifyEqual(string(categories(sentClass)),["positive";"negative";"neutral"]);            
            test.verifyEqual(extractdata(sentScore),single([0.4561, -0.9558]),'AbsTol',single(1e-4));
        end        
    end    
end