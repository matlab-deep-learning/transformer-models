classdef(SharedTestFixtures = {
        DownloadBERTFixture}) tbert < matlab.unittest.TestCase
    % tbert   System level tests for bert
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(TestParameter)
        UncasedVersion = {"base", ...
                   "tiny"}
    end
    
    methods(Test)
        function canConstructModelWithDefault(test)
            % Verify the default model can be constructed.
            test.verifyWarningFree(@() bert());
        end
           
        function canConstructModelWithNVPAndVerifyDefault(test)
            % Verify the default model matches the default model.
            mdl = test.verifyWarningFree(@() bert('Model', "base"));
            mdlDefault = bert();
            test.verifyThat(mdl, iIsEqualTo(mdlDefault));
        end
        
        function checkBertIgnoreCase(test, UncasedVersion)
            % Test that bert() is case insensitive.
            
            txt = "Nipson anomemata. Memonan opsin.";
            
            mdl = bert('Model', UncasedVersion);
            yLower = mdl.Tokenizer.encode(txt);
            yUpper = mdl.Tokenizer.encode(upper(txt));
            
            test.verifyThat(yLower, iIsEqualTo(yUpper));            
        end
        
        function multicasedVersionIsCaseSensitive(test)
            % Check that the multicased version is case sensitive.
            
            txt = "Nipson anomemata. Memonan opsin.";
            
            mdl = bert('Model', "multilingual-cased");
            yLower = mdl.Tokenizer.encode(txt);
            yUpper = mdl.Tokenizer.encode(upper(txt));
            
            test.verifyThat(yLower, iIsNotEqualTo(yUpper));
        end
        
        function canDoNSP(test)
            % Verify the next-sentence prediction works.
            mdl = bert();
            text_a = "MATLAB combines a desktop environment tuned for iterative analysis and design processes with a programming language that expresses matrix and array mathematics directly.";
            text_b_next = "It includes the Live Editor for creating scripts that combine code, output, and formatted text in an executable notebook.";
            text_b_random = "This is just a test, no need to be alarmed.";
            x = mdl.Tokenizer.encode([text_a;text_a],[text_b_next;text_b_random]);
            x = padsequences(x,2,'PaddingValue',mdl.Tokenizer.PaddingCode);
            prediction = bert.layer.classifierHead(bert.model(x,mdl.Parameters),mdl.Parameters.Weights.pooler,mdl.Parameters.Weights.sequence_relation);
            probability = softmax(prediction,'DataFormat','CB');
            classes = onehotdecode(probability,["IsNext","IsNotNext"],1);
            test.verifyEqual(classes,categorical(["IsNext","IsNotNext"]));
        end
    end
end

function constraint = iIsEqualTo(varargin)
constraint = matlab.unittest.constraints.IsEqualTo(varargin{:});
end

function constraint = iIsNotEqualTo(varargin)
constraint = ~matlab.unittest.constraints.IsEqualTo( varargin{:} );
end