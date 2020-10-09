classdef tnormalization < matlab.unittest.TestCase
    % tnormalization   Tests for transformer.layer.normalization
    % The expected behaviour is to normalize the mean to zero and standard
    % deviation to 1, then perform an affine linear transform.
    % Normalization occurs along the 1st dimension.
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties(Constant)
        normalization = @transformer.layer.normalization
        Eps = single(1e-5)
    end
    
    properties(TestParameter)
        Precision = struct( ...
            'double',@double, ...
            'single',@single )
        InputClass = struct( ...
            'identity', @(x) x, ...
            'dlarray', @(x) dlarray(x) )
        Value = struct( ...
            'scalarOne', struct('input',1, 'expected', 0), ...
            'scalarTwo', struct('input',2, 'expected', 0), ...
            'scalarThree', struct('input',0, 'expected', 0), ...
            'vectorSymmetric', struct('input',[-1/sqrt(2);0;1/sqrt(2)],'expected',[-sqrt(3)/sqrt(2);0;sqrt(3)/sqrt(2)]), ...
            'vectorSymmetricTwo', struct('input',[1;2;3],'expected',[-sqrt(3)/sqrt(2);0;sqrt(3)/sqrt(2)]), ...
            'matrix', struct('input',reshape(1:9,[3,3]),'expected',referenceImplementation(reshape(1:9,[3,3]),1,0,1)) )
    end
    
    methods(Test)
        function computesExpectedValue(test,Precision,InputClass,Value)
            x = Precision(Value.input);
            x = InputClass(x);
            y = test.normalization(x,1,0);
            % Because normalization uses single(1e-5) as epsilon it always
            % casts to single.
            if isa(y,'dlarray')
                y = extractdata(y);
            end
            test.verifyEqual(y,single(Value.expected),'AbsTol',2*test.Eps);
        end
    end
end

function y = referenceImplementation(x,g,b,normDim)
mu = mean(x,normDim);
sig = std(x,1,normDim);
xhat = (x-mu)./(sig+eps(class(sig)));
y = g.*xhat + b;
end