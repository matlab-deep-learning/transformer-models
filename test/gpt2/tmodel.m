classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tmodel < matlab.unittest.TestCase
    % tmodel   Tests for gpt2.model
    
    % Copyright 2020 The MathWorks, Inc.
    
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
            parametersFile = fullfile(getRepoRoot(),'gpt2-355M','parameters.mat');
            parameters = gpt2.load(parametersFile);
        end
    end
end

function s = iGetInputData()
s = struct( ...
    'SingleToken', dlarray(1), ...
    'MultiSeqLen', dlarray([1 7 2 9]), ...
    'MultiSeqLenAndObs', dlarray( [1 7 2 9; 7 2 1 9] ) ...
);
end