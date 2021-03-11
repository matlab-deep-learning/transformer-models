classdef(SharedTestFixtures={
        DownloadBERTFixture}) tlanguageModel < matlab.unittest.TestCase
        
    % tlanguageModel   Unit test for bert.languageModel
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(Constant)
        FunctionUnderTest = @(x, p)bert.languageModel(x, p);
    end
    
    methods(Test)
        function canBatch(test)
            mdl = bert('Model','tiny');
            x = dlarray(repmat(1:10,[1,1,2]));
            probs = extractdata(bert.languageModel(x,mdl.Parameters));
            test.verifyEqual(probs(:,:,1),probs(:,:,2),'AbsTol',single(1e-6));
        end
        
        function verifyOutputDimSizes(test)
            % Test that the output dimension sizes of bert.languageModel 
            % have expected values.
            
            txt = ["nipson anomemata"; "memonan opsin"];
            numObs = numel(txt);
            mdl = bert('Model', "base");
            params = mdl.Parameters;
            seqs = mdl.Tokenizer.encode(txt);
            paddingValue = mdl.Tokenizer.PaddingCode;
            seqsPadded = padsequences(seqs, 2, "PaddingValue", paddingValue);
            vocabularySize = size(params.Weights.masked_LM.output.bias, 1);

            x = dlarray(seqsPadded);
            
            z = bert.languageModel(seqsPadded, params);
            
            test.verifyNotEmpty(z);
            test.verifyThat(size(z, 1), iIsEqualTo(vocabularySize), ...
                            'Wrong first output dim size.');
            test.verifyThat(size(z, 2), iIsEqualTo(size(x, 2)), ...
                            'Wrong second output dim size.');            
            test.verifyThat(size(z, 3), iIsEqualTo(numObs), ...
                            'Wrong third output dim size.');
        end
        
        function canPredictMaskedTokens(test)
            % Test that BERT can predict masked tokens.
            
            mdl = bert('Model', 'base');
            str = "Today it is warm and sunny so I have to drink cold water.";
            seq = mdl.Tokenizer.encode(str);
            x = dlarray(seq{1});
            
            % Replace some sequence element with a mask. Specifically the
            % first word before full-stop.
            toMask = size(x, 2) - 2;
            x(toMask) = mdl.Tokenizer.MaskCode;
            z = bert.languageModel(x, mdl.Parameters);
            
            k = 5;
            maskProbabilities = z(:, toMask);
            [~, topk] = maxk(extractdata(maskProbabilities), k);
            toks = arrayfun(@(idx) mdl.Tokenizer.decode(idx), topk);
            
            % Regression test against hard coded values.
            test.verifyThat(toks, iIsEqualTo(["water"; "coffee"; "tea"; "beer"; "milk"]));        
        end
        
        function checkProbDistrOverChannelDimBatches(test)
            % Verify that the output of bert.languageModel corresponds to
            % a probability distribution over the implicit C dimension
            % summing up to 1staking into account a batched input.
            
            mdl = bert('Model', 'tiny');
            x = dlarray(repmat(1:10, [1, 1, 2]));
            z = bert.languageModel(x, mdl.Parameters);
            
            actualProbSum = extractdata(sum(z, 1));
            expectedProbSum = single(ones(1, size(z, 2), 2));
            tolerance = iAbsoluteTolerance(single(1e-6));
            
            test.verifyThat(actualProbSum, ...
                iIsEqualTo(expectedProbSum, 'Within', tolerance));
        end
    end
end

function constraint = iIsEqualTo(varargin)
constraint = matlab.unittest.constraints.IsEqualTo(varargin{:});
end

function constraint = iAbsoluteTolerance(tol)
constraint = matlab.unittest.constraints.AbsoluteTolerance(tol);
end