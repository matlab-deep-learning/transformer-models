classdef(SharedTestFixtures={DownloadFinBERTFixture}) ...
        tlanguageModel < matlab.unittest.TestCase
    % tlanguageModel   Unit test for finbert.languageModel
    
    % Copyright 2021 The MathWorks, Inc.  
    
    properties(Constant)
        FunctionUnderTest = @(x, p)finbert.languageModel(x, p);
    end
    
    methods(Test)
        
        function canBatch(test)
            mdl = finbert('Model', "language-model");
            x = dlarray(repmat(1:10,[1,1,2]));
            probs = extractdata(finbert.languageModel(x,mdl.Parameters));
            test.verifyEqual(probs(:,:,1),probs(:,:,2),'AbsTol',single(1e-6));
        end
        
        function canPredictMaskedTokens(test)
            mdl = finbert('Model','language-model');
            str = "This investment has no return on value.";
            seq = mdl.Tokenizer.encode(str);
            x = dlarray(seq{1});
            % Replace some sequence element with a mask
            toMask = size(x,2)-2;
            x(toMask) = mdl.Tokenizer.MaskCode;
            z = finbert.languageModel(x,mdl.Parameters);
            k = 3;
            maskProbabilities = z(:,toMask);
            [~,topk] = maxk(extractdata(maskProbabilities),k);
            toks = arrayfun(@(idx) mdl.Tokenizer.decode(idx), topk);
            % Regression test against hard coded values.
            test.verifyEqual(toks,["investment";"value";"equity"]);
        end
        
        function languageModelErrorsWithSAParams(test)
            % Test that bert.languageModel errors when using a sentiment
            % analysis input model parameters.
            
            mdl = finbert('Model', 'sentiment-model');
            str = "This investment has no return on value.";    
            lmFunction = @() finbert.languageModel(str, mdl.Parameters);
            
            test.verifyError(lmFunction, 'bert:languageModel:MissingLMWeights');
        end
        
    end
end