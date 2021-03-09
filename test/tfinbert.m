classdef (SharedTestFixtures={
        DownloadFinBERTFixture}) ...
        tfinbert < matlab.unittest.TestCase
    
    
    methods(Test)        
        function canConstructModelWithDefault(test)
            % Verify the default model can be constructed.
            test.verifyWarningFree(@() finbert());
        end
        
        function canConstructModelWithNVP(test)
            % Verify the default model matches the sentiment analysis
            % model.
            mdl = test.verifyWarningFree(@() finbert('Model','sentiment-model'));
            mdlDefault = finbert();
            test.verifyEqual(mdl,mdlDefault);
        end
        
        function canConstructLanguageModel(test)
            % Verify that we can construct the language model
            test.verifyWarningFree(@() finbert('Model','language-model'));
        end        
    end
end