classdef tdropout < matlab.unittest.TestCase
    % tdropout   Unit tests for transformer.layer.dropout
    
    % Copyright 2020 The MathWorks, Inc.
    properties(TestParameter)
        Size = struct(...
            'Vector',[10000,1],...
            'Matrix',[100,100],...
            'Tensor',[5,60,70])
        Probability = struct(...
            'Zero',0,...
            'Half',0.5,...
            'OneIsh',0.95)
    end
    
    methods(TestMethodSetup)
        function resetGlobalSeed(~)
            rng(0);
        end
    end
    
    methods(Test)
        function doesDropout(test,Size,Probability)
            x = ones(Size);
            y = transformer.layer.dropout(x,Probability);
            % Expectation is that mean(y) is about 1 since we use inverted
            % dropout.
            actMean = mean(y,'all');
            expMean = 1;
            tol = 1e-1;
            test.verifyEqual(actMean,expMean,'AbsTol',tol);
        end
        
        function isRandom(test)
            % Verify dropout is random on repeated calls
            x = ones([5,4,3]);
            p = 0.5;
            y1 = transformer.layer.dropout(x,p);
            y2 = transformer.layer.dropout(x,p);
            test.verifyNotEqual(y1,y2);
        end
        
        function supportsDlarrayAndAutodiff(test)
            x = dlarray(ones([5,4,3]));
            p = 0.5;
            f = @(x) transformer.layer.dropout(x,p);
            test.verifyWarningFree(@() f(x));
            function [val,df_val] = df(x)
                val = f(x);
                df_val = dlgradient(sum(val,'all'),x);
            end
            [f_val,df_val] = dlfeval(@df,x);
            % Gradient of dropout is itself
            test.verifyEqual(df_val,f_val);
        end
    end
end