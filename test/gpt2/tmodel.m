classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tmodel < matlab.unittest.TestCase
    % tmodel   Tests for gpt2.model
    
    % Copyright 2020-2021 The MathWorks, Inc.
    
    properties(Constant)
        model = @gpt2.model
    end
    
    properties(TestParameter)
        InputData = iGetInputData();
    end
    
    methods(Test)
        function canUseModel(test, InputData)
            X = InputData;
            [pasts, parameters] = test.prepareInputs();
            test.verifyWarningFree(@() test.model(X, pasts, parameters));
        end
        
        function canAcceptBatches(test)
            % gpt2.model should be able to accept multiple observations
            % with the same sequence length
            
            % Create inputs
            [pasts, parameters] = test.prepareInputs();
            numObs = 4;
            seqLen = 5;
            vocabSize = size( parameters.Weights.wte_0, 2 );
            X = randi(vocabSize, [1 seqLen numObs]);
            
            % Get batch results
            Ybatch = test.model(X, pasts, parameters);
            
            % Iterate over batch
            YperObs = dlarray(zeros([vocabSize seqLen numObs], 'single'));
            for i = 1:numObs
                YperObs(:, :, i) = test.model(X(:, :, i), pasts, parameters);
            end
            
            % Verify the results are within a relative tolerance for single
            % precision data
            test.verifyEqual(extractdata(Ybatch), extractdata(YperObs), 'RelTol', single(1e-5));
        end
    end
    
    methods(Access=private)
        function [pasts, parameters] = prepareInputs(test)
            % Convenience method to setup inputs for
            % transformer.model
            parameters = test.prepareParameters();
            pasts = test.preparePasts(parameters.Hyperparameters.NumLayers);
        end
        
        function pasts = preparePasts(~,numLayers)
            pasts = cell(numLayers,1);
        end
        
        function parameters = prepareParameters(~)
            parametersFile = gpt2.internal.getSupportFilePath("gpt2_355M_params.mat");
            parameters = gpt2.load(parametersFile);
        end
    end
end

function s = iGetInputData()
s = struct( ...
    'SingleToken', dlarray(1), ...
    'MultiSeqLen', dlarray([1 7 2 9]), ...
    'MultiSeqLenAndObs', dlarray( permute([1 7 2 9; 7 2 1 9], [3 2 1]) ) ...
);
end