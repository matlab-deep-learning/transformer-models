classdef(SharedTestFixtures = {
        DownloadBERTFixture}) tBERTTokenizer < matlab.unittest.TestCase
    % tBERTTokenizer   Unit tests for the BERTTokenizer.
    
    % Copyright 2020-2021 The MathWorks, Inc.
    
    properties(TestParameter)
        VocabFiles = iVocabFiles()
    end
    
    properties(Constant)
        Constructor = @bert.tokenizer.BERTTokenizer
    end
    
    methods(Test)
        function canConstruct(test)
            % Note - the tokenizer requires a downloaded model to be
            % constructed. In particular it needs the vocab.txt
            test.verifyWarningFree(@()test.Constructor());
        end
        
        function canConstructWithNonDefaultModel(test)
            % Verify the optional model argument
            tok = test.Constructor();
            tok2 = test.Constructor(bert.internal.getSupportFilePath("base","vocab.txt"));
            % The argument is only used for locating the vocab.txt, and
            % these are the same for every model so far, so the tokenizers
            % are equal.
            test.verifyEqual(tok,tok2);
        end
        
        function hasExpectedProperties(test, VocabFiles)
            % Verify the public gettable properties have the expected
            % values
            tok = test.Constructor(VocabFiles);
            test.verifyEqual(tok.PaddingToken, "[PAD]");
            test.verifyEqual(tok.StartToken, "[CLS]");
            test.verifyEqual(tok.SeparatorToken, "[SEP]");
            % Technically the following can be derived from FullTokenizer -
            % but that would be duplicating what the source code does.
            test.verifyEqual(tok.PaddingCode, 1);
            test.verifyEqual(tok.SeparatorCode, 103);
            test.verifyEqual(tok.StartCode, 102);
        end
        
        function canEncodeOneSentence(test)
            % Test encoding of a single observation
            tok = test.Constructor();
            x = "foo, bar. Baz!";
            y = tok.encode(x);
            test.verifyClass(y,'cell');
            y1 = y{1};
            test.verifyEqual(y1(1),tok.StartCode);
            test.verifyEqual(y1(end),tok.SeparatorCode);
            % Regression test against hard-coded values
            test.verifyEqual(y1(2:end-1),[29380, 1011, 3348, 1013, 8671, 2481, 1000]);
        end
        
        function canEncodeMultipleSentences(test, VocabFiles)
            % Test encoding multiple observations;
            tok = test.Constructor(VocabFiles);
            x = [
                "foo, bar. Baz!"
                "hello world"
                "Bidirectional Encoder Representations from Transformers"
                "Word piece tokenization"];
            % give x an interesting shape
            x = reshape(x,2,1,2);
            act_y = tok.encode(x);
            % Expect this to simply match encoding each observation in
            % turn.
            exp_y = reshape(arrayfun(@(x) tok.encode(x), x), size(x));
            test.verifyEqual(act_y,exp_y);
        end
        
        function canEncodeSentencePair(test, VocabFiles)
            % Test encoding sentence pairs
            tok = test.Constructor(VocabFiles);
            x1 = "foo, bar. Baz!";
            x2 = "hello world";
            act_y = tok.encode(x1,x2);
            % Expected value is to encode x1 and x2 and join on the
            % separator - however we encoding appends and prepends the
            % closing separator and start token, so remember that here.
            y1 = tok.encode(x1);
            y1 = y1{1};
            y2 = tok.encode(x2);
            y2 = y2{1};
            exp_y = {[y1,y2(2:end)]};
            test.verifyEqual(act_y,exp_y);
        end
        
        function canEncodeMultipleSentencePairs(test, VocabFiles)
            % Test encoding multiple sentence pairs
            tok = test.Constructor(VocabFiles);
            x1 = [
                "foo, bar. Baz!"
                "hello world"
                "Bidirectional Encoder Representations from Transformers"
                "Word piece tokenization"];
            % Give x1 an interesting shape
            x1 = reshape(x1,2,1,2);
            x2 = [
                "sentence pair"
                "multiple sentence pairs"
                "BERT model"
                "test"];
            act_y = tok.encode(x1,x2);
            % The expectation is that each pair is encoded in turn
            exp_y = cell(4,1);
            for i = 1:4
                exp_y(i) = tok.encode(x1(i),x2(i));
            end
            % However the shape of the output matches the first input
            exp_y = reshape(exp_y,size(x1));
            test.verifyEqual(act_y,exp_y);
        end
        
        function errorsForDifferentNumberOfSentencePairObservations(test, VocabFiles)
            % Verify an error is thrown when different number of
            % observations are used for sentence pairs
            tok = test.Constructor(VocabFiles);
            x1 = "foo";
            x2 = ["bar","baz"];
            test.verifyError(@() tok.encode(x1,x2), 'bert:tokenizer:SentencePairNumelMismatch');
        end
        
        function canDecode(test)
            % Verify the decode method is "nearly" an inverse of encode -
            % it leaves the special tokens in and assumes space separation.
            x = "foo, bar. Baz!";
            tok = test.Constructor();
            y = tok.encode(x);
            act_decode = tok.decode(y{1});
            exp_decode = "[CLS] foo , bar . ba ##z ! [SEP]";
            test.verifyEqual(act_decode,exp_decode);
        end
        
        function canDecodeMultipleObservations(test, VocabFiles)
            % verify multiple observations can be decoded
            x = [
                "foo, bar. Baz!"
                "hello world"];
            tok = test.Constructor(VocabFiles);
            y = tok.encode(x);
            act_decode = tok.decode(y);
            exp_decode = [tok.decode(y{1});tok.decode(y{2})];
            test.verifyEqual(act_decode,exp_decode);
        end
        
        function canDecodePaddedBatch(test, VocabFiles)
            % decode also supports padded inputs in CTB format
            x = reshape(1:6,1,3,2);
            tok = test.Constructor(VocabFiles);
            act_decode = tok.decode(x);
            exp_decode = [tok.decode(x(:,:,1));tok.decode(x(:,:,2))];
            test.verifyEqual(act_decode,exp_decode);
        end
        
        function canIgnoreCase(test, VocabFiles)
            % Check that BERTTokenizer is case insensitive if IgnoreCase is
            % TRUE.
            
            % Set 'IgnoreCase' equal to TRUE.
            tok = test.Constructor(VocabFiles, 'IgnoreCase', true);
            xLower = "foo, bar, baz!";
            xUpper = upper(xLower);
            
            yLower = tok.encode(xLower);
            yUpper = tok.encode(xUpper);
            
            % Verify that both encodings are equal for both lower and upper
            % case inputs.
            test.verifyThat(yLower, iIsEqualTo(yUpper));
        end
                 
        function checkCaseSensitivity(test, VocabFiles)
            % Check that BERTTokenizer is case sensitive if IgnoreCase is
            % FALSE.
            
            % Set 'IgnoreCase' equal to FALSE.
            tok = test.Constructor(VocabFiles, 'IgnoreCase', false);
            xLower = "foo, bar, baz!";
            xUpper = upper(xLower);
            
            yLower = tok.encode(xLower);
            yUpper = tok.encode(xUpper);
            
            % Verify that encodings are not equal for lower and upper
            % case inputs.
            isNotEqualTo_yUpper = ~iIsEqualTo(yUpper);
            test.verifyThat(yLower, isNotEqualTo_yUpper);
        end
    end    
end

function constraint = iIsEqualTo(varargin)
constraint = matlab.unittest.constraints.IsEqualTo(varargin{:});
end

function vocabFiles = iVocabFiles()
versions = ["base","tiny","mini","small","medium","multilingual-cased"];
vocabFiles = cell(size(versions));
for i = 1:numel(versions)
    vocabFiles{i} = bert.internal.getSupportFilePath(versions(i),"vocab.txt");
end
end    