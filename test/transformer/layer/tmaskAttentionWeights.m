classdef tmaskAttentionWeights < matlab.unittest.TestCase
    % tmaskAttentionWeights   Tests for
    % transformer.layer.maskAttentionWeights
    
    % Copyright 2020 The MathWorks, Inc.
    
    % The expected behaviour is a temporal mask - states a(t) should
    % not attend to future states a(s) for any s>t. Since a softmax is
    % applied after this step it suffices to mask with a large negative
    % value.
    
    properties(Constant,Access=private)
        maskAttentionWeights = @transformer.layer.maskAttentionWeights
    end
    
    properties(TestParameter)
        Input = iInput
    end
    
    methods(Test)
        function hasExpectedValue(test,Input)
            W = dlarray(Input);
            act_masked = test.maskAttentionWeights(W);
            [m,n] = size(W,[1,2]);
            % Reimplement the triu call
            mask = ((m-n)+(1:n))>=(1:m)';
            test.verifyEqual(W(mask),act_masked(mask));
            import matlab.unittest.constraints.EveryElementOf
            import matlab.unittest.constraints.IsLessThan
            act_masked = extractdata(act_masked);
            masked_vals = act_masked(~mask);
            if ~isempty(masked_vals)
                test.verifyThat(EveryElementOf(masked_vals),IsLessThan(-1e9));
            end
        end
    end
end

function s = iInput()
s = struct(...
    'One', 1,...
    'Two', magic(2),...
    'ThreeTwo', rand(3,2),...
    'TwoThree', rand(2,3),...
    'MultiObs', rand(5,4,3));
end