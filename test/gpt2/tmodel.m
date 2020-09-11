classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tmodel < matlab.unittest.TestCase
    % tmodel   Tests for gpt2.model
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties(Constant)
        model = @gpt2.model
    end
    
    methods(Test)
        function canUseModel(test)
            inputs = test.prepareInputs();
            test.verifyWarningFree(@() test.model(inputs{:}));
        end
    end
    
    methods(Access=private)
        function inputs = prepareInputs(test)
            % Convenience method to setup inputs for
            % transformer.model
            X = test.prepareX();            
            parameters = test.prepareParameters();
            pasts = test.preparePasts(parameters.Hyperparameters.NumLayers);
            inputs = {X,pasts,parameters};
        end
        
        function X = prepareX(~)
            X = dlarray(1);
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