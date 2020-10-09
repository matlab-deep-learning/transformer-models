classdef ttopKLogits < matlab.unittest.TestCase
    % ttopKLogits   Unit tests for sampling.topKLogits
    
    % Copyright 2020 The MathWorks, Inc.
    
    methods(Test)
        function testForKIsOne(test)
            % When K=1 topKLogits followed by softmax is just the one-hot
            % for the argmax.
            numClasses = 100;
            % Make an arbitrary class the largest logit.
            class = randi([1,numClasses]);
            x = rand([numClasses,1]);
            x(class) = max(x,[],'all')+1;
            % Call topKLogits with K=1
            topK = 1;
            logits = sampling.topKLogits(x,topK);
            logits = dlarray(logits);
            % Apply softmax to get a probability distribution over the
            % classes.
            actProb = softmax(logits,'DataFormat','C');
            % The expected output is a delta distribution on the arbitrary
            % class.
            expProb = zeros([numClasses,1]);
            expProb(class) = 1;
            test.verifyEqual(extractdata(actProb),expProb);
        end
        
        function testForLargeK(test)
            % When K>1 the topKLogits followed by softmax are only
            % non-zero at the largest K classes. If those classes have
            % equal logits the distribution is uniform.
            numClasses = 100;
            K = 10;
            classes = (1:K)+randi([1,numClasses-K]);
            x = rand([numClasses,1]);
            x(classes) = max(x,[],'all')+1;
            logits = sampling.topKLogits(x,K);
            logits = dlarray(logits);
            actProb = softmax(logits,'DataFormat','C');
            expProb = zeros([numClasses,1]);
            expProb(classes) = 1/K;
            test.verifyEqual(extractdata(actProb),expProb);
        end
        
        function dlarrayIsSupported(test)
            % Test we can call topKLogits on a dlarray.
            numClasses = 10;
            k = 5;
            x = dlarray(rand([numClasses,1]));
            yAct = sampling.topKLogits(x,k);
            yExp = sampling.topKLogits(extractdata(x),k);
            test.verifyEqual(extractdata(yAct),yExp);
        end
    end
    
    
end