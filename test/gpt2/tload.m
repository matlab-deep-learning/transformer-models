classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tload < matlab.unittest.TestCase
    % tload   Test for gpt2.load
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties(Constant)
        ExpectedNumHeads = 16
        ExpectedNumLayers = 24
        ExpectedContext = 1024
    end
    
    properties
        Parameters
    end
    
    methods(TestClassSetup)
        function loadParameters(test)
            % Load the parameters once for all tests
            test.Parameters = gpt2.load(gpt2.internal.getSupportFilePath('gpt2_355M_params.mat'));
        end
    end
    
    methods(Test)
        function verifyLoadStructFields(test)
            % Verify the expected fieldnames of the loaded struct
            import matlab.unittest.constraints.IsSameSetAs
            expected = ["Hyperparameters","Weights"];
            test.verifyThat(fieldnames(test.Parameters), IsSameSetAs(expected));
        end
        
        function verifyHyperparameters(test)
            % Verify the 355M config
            hyperParams = test.Parameters.Hyperparameters;
            test.verifyEqual(hyperParams.NumHeads,test.ExpectedNumHeads,...
                "Unexpected value for Hyperparameters.NumHeads");
            test.verifyEqual(hyperParams.NumLayers,test.ExpectedNumLayers,...
                "Unexpected value for Hyperparameters.NumLayers");
            test.verifyEqual(hyperParams.NumContext,test.ExpectedContext,...
                "Unexpected value for Hyperparameters.NumContext");
        end
        
        function verifyWeights(test)
            % Verify the structure of the Weights field and check some
            % particular weight has the expected type.
            
            % Here there is an implicit check that "model_" has been
            % removed from the weight names and the flat parameters.mat has
            % been organised into a heirarchy for each gpt2.block
            w = test.assertWarningFree(@() test.Parameters.Weights.h0.ln_1_g_0);
            import matlab.unittest.constraints.IsOfClass
            test.verifyThat(w,IsOfClass('dlarray'));
            test.verifyThat(extractdata(w),IsOfClass('single'));
        end
    end    
end