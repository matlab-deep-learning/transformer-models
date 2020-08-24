classdef tgelu < matlab.unittest.TestCase
    % tgelu   Tests for transformer.layer.gelu
    
    % Copyright 2020 The MathWorks, Inc.
    
    % Reference: https://arxiv.org/abs/1606.08415
    
    properties(Constant)
        gelu = @transformer.layer.gelu
        Tolerance = 3e-4
    end
    
    properties(TestParameter)
        Value = {0,1,-5,10,1.23,100}
        Cast = struct(...
            'double', @double, ...
            'single', @single, ...
            'dlarray', @dlarray)
    end
    
    methods(Test)
        function matchesExpectedValue(test,Value)
            % Check gelu matches the non-approximate version within some
            % tolerance.
            actual = test.gelu(Value);
            expected = Value * normcdf(Value);
            tol = test.mixedTolerance();
            constraint = matlab.unittest.constraints.IsEqualTo(expected,'Within',tol);
            test.verifyThat(actual,constraint);
        end
        
        function supportsMultipleTypes(test,Cast)
            % Check gelu supports various input types and has the expected
            % value.
            num = pi;
            casted = Cast(num);
            expected = test.gelu(num);
            actual = test.gelu(casted);
            test.verifyEqual(actual,Cast(expected));
        end
        
    end
    
    methods(Access=private)
        function tol = mixedTolerance(test)
            absTol = matlab.unittest.constraints.AbsoluteTolerance(test.Tolerance);
            relTol = matlab.unittest.constraints.RelativeTolerance(test.Tolerance);
            tol = absTol | relTol;
        end
    end
end

function y = normcdf(x)
% reimplement normcdf for mean 0 std 1
y = (1+erf(x/sqrt(2)))/2;
end