classdef tattention < matlab.unittest.TestCase
    % tattention   Tests for transformer.layer.attention
    
    % Copyright 2020 The Mathworks, Inc.
    
    % The attention function has 3 purposes:
    % 1. Apply the pre and post attention fully connected layers.
    % 2. Rearrange data format for applying multi-head attention.
    % 3. Preserve the key and value matrices, and append to them when using
    % values from the past. This is done pre-attention.
    
    properties(Constant,Access=private)
        attention = @transformer.layer.attention
        multiheadAttention = @transformer.layer.multiheadAttention
    end
    
    properties(TestParameter)
        NumQueries = {1,2}
        NumObs = {1,3}
    end
    
    methods(Test)
        function checkSingleHeadNoFullyConnected(test,NumQueries,NumObs)
            % Verify that multiheadAttention is used.
            % Do this by setting the fully connected weights to identity
            % and the bias to 0.
            % With a single head we do not need split and merge heads.
            latentDim = 10;
            hyperParams.NumHeads = 1;
            past = [];
            % Set up the fc weights to be identity matrices and biases to
            % be 0.
            weights = test.prepareWeightsStructWithIdentityFC(hyperParams.NumHeads,latentDim);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*3,NumQueries,NumObs));
            yAct = test.attention(x,past,weights,hyperParams);
            % Verify against multiheadAttention by splitting the arbitrary
            % input into query, key and value.
            [q,k,v] = iSplitQKV(x,hyperParams.NumHeads,latentDim);
            yExp = test.multiheadAttention(q,k,v);
            % Note that for numObs > 1 we need to merge heads. Since we
            % have a single head here, no merger needs to take place -- we
            % just need to permute the observation dimension from 4 to 3
            yExp = permute(yExp, [1 2 4 3]);
            test.verifyEqual(yAct,yExp);
        end
        
        function checkMultiHeadNoFullyConnected(test,NumQueries,NumObs)
            % Verify the multiple head case. Multi head attention is
            % achieved by simply creating a "head dimension" and pushing it
            % to the back so that it is treated as a "batch"
            % dimension for matrix multiplication.
            latentDim = 10;
            hyperParams.NumHeads = 2;
            past = [];
            % Set up the fc weights to be identity matrices and biases to
            % be 0. 
            weights = test.prepareWeightsStructWithIdentityFC(hyperParams.NumHeads,latentDim);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,NumQueries,NumObs));
            yAct = test.attention(x,past,weights,hyperParams);
            [Q,K,V] = iSplitQKV(x,hyperParams.NumHeads,latentDim);
            % Verify against the multiheadAttention function.
            yExp = test.multiheadAttention(Q,K,V);
            yExp = iMergeHeads(yExp);
            test.verifyEqual(yAct,yExp);
        end
        
        function checkPastPresentCaching(test,NumQueries,NumObs)
            % Verify the 2nd input and output of attention - the pasts
            % passed in are the keys and values for the previous time step.
            % These are concatenated to the keys and values for the current
            % time step before the multiheadAttention call, and these
            % concatenated key and values are passed out as a second
            % output.
            latentDim = 10;
            hyperParams.NumHeads = 2;
            % Set up a fake past by making an initial call to attention.
            past = [];
            % Set up the fc weights to be identity matrices and biases to
            % be 0. 
            weights = test.prepareWeightsStructWithIdentityFC(hyperParams.NumHeads,latentDim);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,NumQueries,NumObs));
            [~,past] = test.attention(x,past,weights,hyperParams);
            % Verify the expected value of past - it is the key and values
            % concatenated on the 4th dimension.
            [~,K,V] = iSplitQKV(x,hyperParams.NumHeads,latentDim);
            test.verifyEqual(past,cat(5,K,V));
            % Now verify second call to attention is possible with the first 
            % past as input - and verify the value of the attention output.
            [yAct,present] = test.attention(x,past,weights,hyperParams);
            [Q,K,V] = iSplitQKV(x,hyperParams.NumHeads,latentDim);
            % Verify the correct value for present.
            pastK = past(:,:,:,:,1);
            pastV = past(:,:,:,:,2);
            test.verifyEqual(extractdata(present),extractdata(cat(5,cat(2,pastK,K),cat(2,pastV,V))),'AbsTol',1e-5);
            % To compute the expected value, concatenate the pasts
            K = cat(2,K,pastK);
            V = cat(2,V,pastV);
            yExp = test.multiheadAttention(Q,K,V);
            yExp = iMergeHeads(yExp);
            test.verifyEqual(extractdata(yAct),extractdata(yExp),'AbsTol',1e-5);
        end
        
        function checkInputOutputFC(test,NumQueries,NumObs)
            % The tests above ensure multiheadAttention is used by
            % attention. Now verify the input and output FC operations are
            % used.
            latentDim = 10;
            hyperParams.NumHeads = 2;
            % Set up a fake past by making an initial call to attention.
            past = [];
            params = test.prepareWeightsStructUsingGenerators(hyperParams.NumHeads,latentDim,@randn,@randn);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,NumQueries,NumObs));
            yAct = test.attention(x,past,params,hyperParams);
            % Verify this matches the full attention implementation
            z = fullyconnect(x,params.attn_c_attn_w_0,params.attn_c_attn_b_0,'DataFormat','CBT');
            [Q,K,V] = iSplitQKV(z,hyperParams.NumHeads,latentDim);
            z = test.multiheadAttention(Q,K,V);
            z = iMergeHeads(z);
            yExp = fullyconnect(z,params.attn_c_proj_w_0,params.attn_c_proj_b_0,'DataFormat','CBT');
            test.verifyEqual(extractdata(yAct),extractdata(yExp),'AbsTol',1e-5);
        end
        
        function defaultIsMaksed(test)
            % Verify the default for the 'CausalMask' NVP is true
            latentDim = 10;
            hyperParams.NumHeads = 2;
            past = [];
            numQueries = 2;
            % Set up the fc weights to be identity matrices and biases to
            % be 0. 
            weights = test.prepareWeightsStructWithIdentityFC(hyperParams.NumHeads,latentDim);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,numQueries));
            default = test.attention(x,past,weights,hyperParams);
            masked = test.attention(x,past,weights,hyperParams,'CausalMask',true);
            test.verifyEqual(extractdata(default),extractdata(masked));            
        end
        
        function canTurnOffMask(test)
            % Verify the 'CausalMask' NVP can be set to false - the expectation
            % is this simply sets 'Masked' to false for the
            % multiheadAttention call.
            latentDim = 10;
            hyperParams.NumHeads = 2;
            past = [];
            numQueries = 2;
            % Set up the fc weights to be identity matrices and biases to
            % be 0. 
            weights = test.prepareWeightsStructWithIdentityFC(hyperParams.NumHeads,latentDim);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,numQueries));
            yAct = test.attention(x,past,weights,hyperParams,'CausalMask',false);
            [Q,K,V] = iSplitQKV(x,hyperParams.NumHeads,latentDim);
            % Verify against the multiheadAttention function with 'Masked'
            % set to false.
            yExp = test.multiheadAttention(Q,K,V,'CausalMask',false);
            yExp = iMergeHeads(yExp);
            test.verifyEqual(yAct,yExp);
        end
        
        function canDropout(test)
            % Verify dropout can be applied to attention - the expectation
            % is that this matches simply setting the Dropout NVP in the
            % multiheadAttention call
            latentDim = 10;
            hyperParams.NumHeads = 2;
            past = [];
            numQueries = 2;
            % Set up the fc weights to be identity matrices and biases to
            % be 0. 
            weights = test.prepareWeightsStructWithIdentityFC(hyperParams.NumHeads,latentDim);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,numQueries));
            p = 0.5;
            % Set global rng between non-deterministic calls.
            rng(0);
            yAct = test.attention(x,past,weights,hyperParams,'Dropout',p);
            [Q,K,V] = iSplitQKV(x,hyperParams.NumHeads,latentDim);
            % Verify against the multiheadAttention function with 'Dropout'
            % set to p
            % Set global rng between non-deterministic calls.
            rng(0);
            yExp = test.multiheadAttention(Q,K,V,'Dropout',p);
            yExp = iMergeHeads(yExp);
            test.verifyEqual(yAct,yExp);
        end
        
        function defaultIsNoDropout(test)
            % Verify the default Dropout is 0.
            latentDim = 10;
            hyperParams.NumHeads = 2;
            numQueries = 2;
            % Set up a fake past by making an initial call to attention.
            past = [];
            params = test.prepareWeightsStructUsingGenerators(hyperParams.NumHeads,latentDim,@randn,@randn);
            % Call attention on an arbitrary input.
            x = dlarray(rand(latentDim*hyperParams.NumHeads*3,numQueries));
            yDefault = test.attention(x,past,params,hyperParams);
            yNoDropout = test.attention(x,past,params,hyperParams,'Dropout',0);
            test.verifyEqual(yDefault,yNoDropout);
        end
    end
    
    methods(Access=private)
        function s = prepareWeightsStruct(~,W1,b1,W2,b2)
            % Prepare a struct compatible with the weights input of
            % attention. These are for the fully connected layers.
            s = struct(...
                'attn_c_attn_w_0',W1,...
                'attn_c_attn_b_0',b1,...
                'attn_c_proj_w_0',W2,...
                'attn_c_proj_b_0',b2);
        end
        
        function s = prepareWeightsStructUsingGenerators(test,nHeads,latentDim,weightGenerator,biasGenerator)
            % Use function handles weightGenerator and biasGenerator to
            % create weight and bias values for the fc layers in attention.
            % The expectation is these function handles map a size vector
            % to an array of that size.
            
            % The query, key and value vectors are flattened into 1
            % dimension for the input.
            % There are nHeads of each, and each is a latentDim
            % dimensional vector   
            inputDim = 3*nHeads*latentDim;
            W1 = dlarray(weightGenerator([inputDim,inputDim]));
            b1 = dlarray(biasGenerator([inputDim,1]));
            
            % For the output we don't need the 3 - the multiheadAttention
            % has taken an expectation of the value vectors under some
            % probability distribution.            
            outputDim = nHeads*latentDim;
            W2 = dlarray(weightGenerator([outputDim,outputDim]));
            b2 = dlarray(biasGenerator([outputDim,1]));
            s = test.prepareWeightsStruct(W1,b1,W2,b2);
        end
        
        function s = prepareWeightsStructWithIdentityFC(test,nHeads,latentDim)
            % Prepare a struct compatible with the weights input of
            % attention such that the fc layers are identity operations.
            s = test.prepareWeightsStructUsingGenerators(nHeads,latentDim,@eye,@zeros);
        end
    end
end

function [Q,K,V] = iSplitQKV(X, nHeads, latentDim)
% Split an input array X into the corresponding Q, K and V arrays.
splitSize = latentDim*nHeads;
Q = iSplitHeads(X(1:splitSize,:,:),splitSize,nHeads);
K = iSplitHeads(X((splitSize+1):2*splitSize,:,:),splitSize,nHeads);
V = iSplitHeads(X((2*splitSize+1):3*splitSize,:,:),splitSize,nHeads);
end

function X = iSplitHeads(X, splitSize, numHeads)
X = reshape(X, splitSize/numHeads, numHeads, [], size(X,3));   % Split states
X = permute(X,[1 3 2 4]);
end

function X = iMergeHeads(X)
X = permute(X, [1 3 2 4]);
X = reshape(X, size(X,1)*size(X,2), [], size(X,4));            % Merge states
end