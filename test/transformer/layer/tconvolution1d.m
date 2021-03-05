classdef tconvolution1d < matlab.unittest.TestCase
    % tconvolution1d   Tests for the convolution1d function
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties(Constant,Access=private)
        convolution1d = @transformer.layer.convolution1d
    end
    
    properties(TestParameter)
        Input = iInput
    end
    
    methods(Test)
        function hasExpectedValue(test,Input)
            x = dlarray(Input.x);
            W = dlarray(Input.W);
            b = dlarray(Input.b);
            z_exp = fullyconnect(x,W,b,'DataFormat','CBT');
            z_act = test.convolution1d(x,W,b);
            test.verifyEqual(extractdata(z_act), extractdata(z_exp), 'AbsTol', 1e-10);
        end
    end
end

function s = iInput()
s = struct(...
    'OneC',iInputCase(1,2,0),...
    'TwoC',iInputCase(ones(2,1),rand(3,2),rand(3,1)),...
    'CT',iInputCase(rand(3,4),rand(5,3),rand(5,1)), ...
    'CBT',iInputCase(rand(3,4,2),rand(5,3),rand(5,1)));
end

function s = iInputCase(x,w,b)
s = struct(...
    'x',x,...
    'W',w,...
    'b',b);
end