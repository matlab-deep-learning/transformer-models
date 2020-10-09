classdef tmultiLayerPerceptron < matlab.unittest.TestCase
    % tmultiLayerPerceptron   Tests for the
    % transformer.layer.multiLayerPerceptron function
    
    % Copyright 2020 The MathWorks, Inc.
    
    % The feed-forward network used in a transformer originally has 2
    % layers with ReLU activation https://arxiv.org/abs/1706.03762
    % However gpt-2 https://openai.com/blog/better-language-models/
    % modifies this to use GeLU activation https://arxiv.org/abs/1606.08415
    
    properties(Constant,Access=private)
        Activation = @transformer.layer.gelu
        multiLayerPerceptron = @transformer.layer.multiLayerPerceptron
    end
    
    properties(TestParameter)
        Input = iInput
    end
    
    methods(Test)
        function hasExpectedValue(test,Input)
            x = dlarray(Input.x);
            W1 = dlarray(Input.W1);
            b1 = dlarray(Input.b1);
            W2 = dlarray(Input.W2);
            b2 = dlarray(Input.b2);
            z_exp = test.expectedValue(x,W1,b1,W2,b2);
            s = test.weightsToStruct(W1,b1,W2,b2);
            z_act = test.multiLayerPerceptron(x,s);
            test.verifyEqual(z_act,z_exp);
        end
    end
    
    methods(Access=private)
        function z = expectedValue(test,x,W1,b1,W2,b2)
            % Computes the expected value of multiLayerPerceptron using Wi
            % and bi as the weights and bias of the i-th fully-connected
            % layer.
            z = fullyconnect(x,W1,b1,'DataFormat','CB');
            z = test.Activation(z);
            z = fullyconnect(z,W2,b2,'DataFormat','CB');
        end
        
        function s = weightsToStruct(~,W1,b1,W2,b2)
            % Create a struct of weights to be consumed by
            % transformer.layer.multiLayerPerceptron
            s = struct(...
                'mlp_c_fc_w_0',W1,...
                'mlp_c_fc_b_0',b1,...
                'mlp_c_proj_w_0',W2,...
                'mlp_c_proj_b_0',b2);
        end
    end
end

function s = iInput()
s = struct( ...
    'OneD', iInputCase(1,1,0,1,0),...
    'TwoD', iInputCase([1;2],eye(2),zeros(2,1),eye(2),zeros(2,1)),...
    'Random', iInputCase(rand([10,1]),rand(15,10),rand(15,1),rand(10,15),rand(10,1)));
end

function s = iInputCase(x,w1,b1,w2,b2)
s = struct(...
    'x',x,...
    'W1',w1,...
    'b1',b1,...
    'W2',w2,...
    'b2',b2);
end