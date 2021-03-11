classdef tmultiheadAttention < matlab.unittest.TestCase
    % tmultiheadAttention   Tests for transformer.layer.multiheadAttention
    
    % Copyright 2020 The MathWorks, Inc.
    
    % Multi-head attention (https://arxiv.org/abs/1706.03762) simply
    % computes a scaled dot product between a query-vector q against a
    % number of keys k1,k2,... that are batched into a matrix K.
    %
    % These projections of the query onto the key vectors are converted to
    % a probability distribution over the keys via softmax.
    %
    % Finally the output is the expected value according to this
    % distribution of the values v1,v2,... associated to keys k1,k2,...
    % which are also typically represented as a matrix V.
    %
    % A batch of queries, q1,q2,... can be merged into a query matrix Q and
    % the whole operation becomes a matrix product.
    %
    % Masking can be applied to prevent the predictions at time t depending
    % on inputs at times s > t.
    
    properties(Constant,Access=private)
        multiheadAttention = @transformer.layer.multiheadAttention
        Tolerance = 1e-6
    end
    
    properties(TestParameter)
        Dim = struct(...
            'scalar',1,...
            'multi',6)
    end
    
    methods(Test)
        function returnsValueWhenOneKey(test)
            % Since softmax is applied over the channel dimension (key
            % dimension), if there is only one key we always expect the
            % result of the softmax to be 1, and so the multiheadAttention
            % returns the value input.
            keyDim = 10;
            numKeys = 1;
            q = dlarray(rand(keyDim,1));
            k = dlarray(rand(keyDim,numKeys));
            v = dlarray(rand);
            attnAct = test.multiheadAttention(q,k,v);
            test.verifyEqual(attnAct,v);
        end
        
        function isScaledDotProduct(test)
            % Verify that multiheadAttention uses the scaled dot-product.
            % This can be seen by setting the value matrix to the identity
            % and verifying the output is the softmax of scaled dot product
            % attention.
            keyDim = 5;
            numKeys = 3;
            q = dlarray(rand(keyDim,1));
            K = dlarray(rand(keyDim,numKeys));
            V = dlarray(eye(numKeys));
            attnAct = test.multiheadAttention(q,K,V);
            attnExp = softmax(test.scaledDotProduct(q,K),'DataFormat','CT');
            test.verifyDlarrayEqual(attnAct,attnExp,'AbsTol',test.Tolerance);
        end
        
        function isExpectedValue(test,Dim)
            % Verify that multiheadAttention is the expected value of the
            % value matrix V over the distribution given by scaled dot
            % product attention with softmax.
            % Since isScaledDotProduct ensures the scaledDotProduct method
            % is used, we can apply softmax to that and compute the
            % expected value directly.
            keyDim = 4;
            numKeys = 5;
            valDim = Dim;
            q = dlarray(rand(keyDim,1));
            K = dlarray(rand(keyDim,numKeys));
            V = dlarray(rand(valDim,numKeys));
            attnAct = test.multiheadAttention(q,K,V);
            attnProbs = softmax(test.scaledDotProduct(q,K),'DataFormat','CT');
            attnExp = test.expectedValue(attnProbs,V);
            test.verifyDlarrayEqual(attnAct,attnExp,'AbsTol',test.Tolerance);
        end
        
        function multipleQueries(test)
            % Verify that multiheadAttention works on multiple queries.
            % This is almost independently, except for masking.
            keyDim = 6;
            numKeys = 7;
            numQueries = 3;
            Q = dlarray(rand(keyDim,numQueries));
            K = dlarray(rand(keyDim,numKeys));
            % Use V = Identity to get the probabilities out.
            V = dlarray(eye(numKeys));
            attnAct = test.multiheadAttention(Q,K,V);
            % Only the last query can attend to every value, then each
            % preceding query can attend to one less value.
            attnScores = test.scaledDotProduct(Q,K);
            for i = 0:(numQueries-1)
                attnExp = softmax(attnScores(1:(end-i),numQueries-i),'DataFormat','CT');
                attnExp = cat(1,attnExp,dlarray(zeros(i,1)));
                test.verifyDlarrayEqual(attnAct(:,numQueries-i),attnExp,'AbsTol',test.Tolerance);
            end
        end
        
        function multipleHeads(test)
            % Verify multiheadAttention works with multiple attention heads
            keyDim = 3;
            valDim = 7;
            numKeys = 4;
            numQueries = 5;
            numHeads = 6;
            Q = dlarray(rand(keyDim,numQueries,numHeads));
            K = dlarray(rand(keyDim,numKeys,numHeads));
            V = dlarray(rand(valDim,numKeys,numHeads));
            attnAct = test.multiheadAttention(Q,K,V);
            attnExp = dlarray(zeros(valDim,numQueries,numHeads));
            for h = 1:numHeads
                attnScores = test.scaledDotProduct(Q(:,:,h),K(:,:,h));
                for i = 0:(numQueries-1)
                    maskedAttnScores = attnScores(1:(end-i),numQueries-i);
                    maskedAttnScores = cat(1,maskedAttnScores,-1e10*dlarray(ones(i,1)));
                    probs = softmax(maskedAttnScores,'DataFormat','CT');
                    attnExp(:,numQueries-i,h) = V(:,:,h)*probs;                    
                end
            end
            test.verifyDlarrayEqual(attnAct,attnExp,'AbsTol',test.Tolerance);
        end        

        function defaultIsMasked(test)
            % Verify the 'CausalMask' NVP
            keyDim = 3;
            valDim = 7;
            numKeys = 4;
            numQueries = 5;
            batchSize = 6;
            Q = dlarray(rand(keyDim,numQueries,batchSize));
            K = dlarray(rand(keyDim,numKeys,batchSize));
            V = dlarray(rand(valDim,numKeys,batchSize));
            default = test.multiheadAttention(Q,K,V);
            masked = test.multiheadAttention(Q,K,V,'CausalMask',true);
            test.verifyDlarrayEqual(default,masked)
        end
        
        function canTurnOffMask(test)
            % Verify the 'CausalMask' NVP can be false - no value should be
            % "very small"
            keyDim = 3;
            valDim = 7;
            numKeys = 4;
            numQueries = 5;
            batchSize = 6;
            Q = dlarray(rand(keyDim,numQueries,batchSize));
            K = dlarray(rand(keyDim,numKeys,batchSize));
            V = dlarray(rand(valDim,numKeys,batchSize));
            A = test.multiheadAttention(Q,K,V,'CausalMask',false);
            import matlab.unittest.constraints.EveryElementOf
            import matlab.unittest.constraints.IsGreaterThan
            test.verifyThat(EveryElementOf(extractdata(A)),IsGreaterThan(-1e-9));
        end
        
        function canDropout(test)
            % By setting the value matrix V to the identity, the output of
            % multiheadAttention with dropout is simply the output without
            % dropout with dropout applied separately.
            keyDim = 5;
            numKeys = 3;
            q = dlarray(rand(keyDim,1));
            K = dlarray(rand(keyDim,numKeys));
            V = dlarray(eye(numKeys));
            attnNoDropout = test.multiheadAttention(q,K,V,'Dropout',0);
            % Reset global seed between non-deterministic calls
            p = 0.5;
            rng(0);
            attnWithDropout = test.multiheadAttention(q,K,V,'Dropout',p);
            rng(0);
            attnExp = transformer.layer.dropout(attnNoDropout,p);
            test.verifyEqual(attnWithDropout,attnExp);
        end
        
        function defaultIsNoDropout(test)
            % Verify the default is no dropout applied to
            % multiheadAttention
            keyDim = 3;
            valDim = 7;
            numKeys = 4;
            numQueries = 5;
            batchSize = 6;
            Q = dlarray(rand(keyDim,numQueries,batchSize));
            K = dlarray(rand(keyDim,numKeys,batchSize));
            V = dlarray(rand(valDim,numKeys,batchSize));
            A_default = test.multiheadAttention(Q,K,V);
            A_exp = test.multiheadAttention(Q,K,V,'Dropout',0);
            test.verifyEqual(A_default,A_exp);
        end
        
        function multipleObservations(test)
            % Verify multiheadAttention works with batches, i.e. across
            % independent observations in a single batch
            keyDim = 3;
            valDim = 7;
            numKeys = 4;
            numQueries = 5;
            numHeads = 6;
            numObs = 2;
            Q = dlarray(rand(keyDim,numQueries,numHeads,numObs));
            K = dlarray(rand(keyDim,numKeys,numHeads,numObs));
            V = dlarray(rand(valDim,numKeys,numHeads,numObs));
            attnAct = test.multiheadAttention(Q,K,V);
            attnExp = dlarray(zeros(valDim,numQueries,numHeads,numObs));
            for n = 1:numObs
                attnExp(:,:,:,n) = test.multiheadAttention(Q(:,:,:,n),K(:,:,:,n),V(:,:,:,n));
            end
            test.verifyDlarrayEqual(attnAct,attnExp,'AbsTol',test.Tolerance);
        end
    end
    
    methods(Access=private)
        function w = scaledDotProduct(~,q,k)
            % The scaled dot product attention for vector or matrices q
            % and k.
            keyDim = size(q,1);
            w = (k'*q)./sqrt(keyDim);
        end
        
        function a = expectedValue(~,p,V)
            % Expected value in this context is simply the weighted sum of
            % the value vectors (columns of V) with weights p.
            % This is the same as a matrix multiply.
            a = V*p;
        end
        
        function verifyDlarrayEqual(test,x,y,varargin)
            test.verifyClass(x,'dlarray');
            test.verifyClass(y,'dlarray');
            test.verifyEqual(extractdata(x),extractdata(y),varargin{:});
        end
    end
end