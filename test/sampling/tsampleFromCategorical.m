classdef tsampleFromCategorical < matlab.unittest.TestCase
    % tsampleFromCategorical   Unit tests for
    % sampling.sampleFromCategorical
    
    % Copyright 2020 The MathWorks, Inc.
    
    methods(Test)
        function canSampleDeterministic(test)
            numClasses = 100;
            class = randi(numClasses);
            probabilities = zeros([numClasses,1]);
            probabilities(class) = 1;
            sample = sampling.sampleFromCategorical(probabilities);
            test.verifyEqual(sample,class);
        end
        
        function canSampleUniform(test)
            numClasses = 10;
            numSamples = 1000;
            uniformProbabilities = ones([numClasses,1])./numClasses;
            samples = arrayfun(...
                @(i) sampling.sampleFromCategorical(uniformProbabilities),1:numSamples);
            counts = groupcounts(samples');
            % Verify the counts for each class are about right
            expMean = numSamples/numClasses;
            tol = 2;
            test.verifyEqual(mean(counts),expMean,'AbsTol',tol);
        end
    end
end