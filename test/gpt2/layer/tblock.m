classdef tblock < matlab.unittest.TestCase
    % tblock   Unit tests for transformer.layer.block
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties(Constant, Access=private)
        block = @gpt2.layer.block
    end
    
    properties(TestParameter)
        Input = struct(...
            'Scalar', 1,...
            'Vector', 1:5,...
            'Matrix', reshape(1:6,[3,2]))
    end
    
    methods(Test)
        function outputHasInputSize(test,Input)
            % The block is simply a composition of other layers. Simply
            % verify the output of a block is the same size as the input,
            % and as such the blocks can be stacked.
            x = dlarray(Input);
            C = size(Input,1);
            weights = test.randomWeights(C);
            hyperParameters.NumHeads = 1;
            y = test.block(x,[],weights,hyperParameters);
            test.verifySize(y,size(x));
        end
        
        function outputHasInputSizeWithPasts(test,Input)
            % As above but using "pasts" - a concatenation of key and value
            % matrices.
            x = dlarray(Input);
            C = size(Input,1);
            weights = test.randomWeights(C);
            hyperParameters.NumHeads = 1;
            % Provide a fake past of sequence length 1
            K_fake = dlarray(rand(C,1));
            V_fake = dlarray(rand(C,1));
            past = cat(5,K_fake,V_fake);
            [y,present] = test.block(x,past,weights,hyperParameters);
            test.verifySize(y,size(x));
            % The size of presents is the size of past except the sequence
            % dimension gets extended by the sequence length of y
            exp_present_size = size(past);
            exp_present_size(2) = exp_present_size(2)+size(y,2);
            test.verifySize(present,exp_present_size);
        end
    end
    
    methods(Access=private)
        function weights = randomWeights(test,C)
            % C is num features, or latent dimension of the block
            g1 = dlarray(rand(C,1));
            b1 = dlarray(rand(C,1));
            g2 = dlarray(rand(C,1));
            b2 = dlarray(rand(C,1));
            W_A1 = dlarray(rand(3*C,C));
            W_A2 = dlarray(rand(C));
            b_A1 = dlarray(rand(3*C,1));
            b_A2 = dlarray(rand(C,1));
            W_P1 = dlarray(rand(C));
            b_P1 = dlarray(rand(C,1));
            W_P2 = dlarray(rand(C));
            b_P2 = dlarray(rand(C,1));
            weights = test.prepareBlockWeightsStruct(g1,b1,W_A1,b_A1,W_A2,b_A2,g2,b2,W_P1,b_P1,W_P2,b_P2);
        end
        
        function s = prepareBlockWeightsStruct(test,g1,b1,W_A1,b_A1,W_A2,b_A2,g2,b2,W_P1,b_P1,W_P2,b_P2)
            % Merge various structs that have the appropriate weight naming
            % syntax.
            s_ln = test.prepareLayerNormWeightsStruct(g1,b1,g2,b2);
            s_attn = test.prepareAttentionWeightsStruct(W_A1,b_A1,W_A2,b_A2);
            s_mlp = test.prepareMLPWeightsStruct(W_P1,b_P1,W_P2,b_P2);
            c = {s_ln,s_attn,s_mlp};
            fn = cellfun(@fieldnames,c,'UniformOutput',false);
            fn = cat(1,fn{:});
            fv = cellfun(@struct2cell,c,'UniformOutput',false);
            fv = cat(1,fv{:});
            s = struct();
            for i = 1:numel(fn)
                s.(fn{i}) = fv{i};
            end
        end
        
        function s = prepareAttentionWeightsStruct(~,W1,b1,W2,b2)
            % Prepare a struct compatible with the weights input of
            % attention. These are for the fully connected layers.
            s = struct(...
                'attn_c_attn_w_0',W1,...
                'attn_c_attn_b_0',b1,...
                'attn_c_proj_w_0',W2,...
                'attn_c_proj_b_0',b2);
        end
        
        function s = prepareLayerNormWeightsStruct(~,g1,b1,g2,b2)
            % Prepare a struct of weights compatible with the two layer
            % norm calls in block
            s = struct(...
                'ln_1_g_0',g1,...
                'ln_1_b_0',b1,...
                'ln_2_g_0',g2,...
                'ln_2_b_0',b2);
        end
        
        function s = prepareMLPWeightsStruct(~,W1,b1,W2,b2)
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