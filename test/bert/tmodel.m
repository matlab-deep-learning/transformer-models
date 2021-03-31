classdef(SharedTestFixtures = {
        DownloadBERTFixture}) tmodel < matlab.unittest.TestCase
    % Unit tests for bert.model
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(TestParameter)
        InvalidOutputs = iInvalidOutputs()
        InvalidCodes = iInvalidCodes()
        InvalidDropoutProb = iInvalidDropoutProb()
        InvalidInputMask = iInvalidInputMask()
        InvalidInputX = iInvalidInputX()
        InvalidModelParams = iInvalidModelParams()
        InvalidModelName = iInvalidModelName()
    end
    
    methods(Test)
        function matchesExpectedValue(test)
            x = dlarray(fake_input_ids());
            params = bert();
            params = params.Parameters;
            % Regression test against some expected values
            v1 = -0.4484;
            vend = 0.0752;
            vmiddle = 0.3042;
            z = bert.model(x,params);
            z = extractdata(z);
            z1 = z(1);
            zend = z(end);
            zmiddle = z(numel(z)/2);
            tol = single(7e-3);
            test.verifyEqual(z1,single(v1),'AbsTol',tol);
            test.verifyEqual(zend,single(vend),'AbsTol',tol);
            test.verifyEqual(zmiddle,single(vmiddle),'AbsTol',tol);
        end
        
        function canBatch(test)
            % Check that BERT returns the same result when either given
            % a batch of sequences or sequence by sequence.
            mdl = bert();
            params = mdl.Parameters;
            txt = ["nipson anomemata"; "memonan opsin"];
            seqs = mdl.Tokenizer.encode(txt);
            paddingValue = mdl.Tokenizer.PaddingCode;
            seqsPadded = padsequences(seqs, 2, "PaddingValue", paddingValue);
            
            x = dlarray(seqsPadded);
            x1 = x(:, :, 1);
            x2 = x(:, :, 2);
            
            y1 = bert.model(x1, params);
            y2 = bert.model(x2, params);
            y = test.verifyWarningFree(@() bert.model(x, params));
            
            tolerance = iAbsoluteTolerance(single(1e-5));
            test.verifyThat(cat(3, extractdata(y1), extractdata(y2)), ...
                iIsEqualTo(extractdata(y), 'Within', tolerance));
        end
        
        function canEncodeSentencePairs(test)
            % Test that multiple sequence pairs pass gives the same results
            % with single sequence pair passes.
            
            txt_a = ["blah blah"; "foo bar baz"];
            txt_b = ["hello world"; "another nonsense string"];
            
            mdl = bert();
            seqs = mdl.Tokenizer.encode(txt_a, txt_b);
            
            paddingValue = mdl.Tokenizer.PaddingCode;
            x = dlarray(padsequences(seqs, 2, 'PaddingValue', paddingValue));
            
            y = bert.model(x, mdl.Parameters);
            y1 = bert.model(x(:, :, 1), mdl.Parameters);
            y2 = bert.model(x(:, :, 2), mdl.Parameters);
            
            tol = matlab.unittest.constraints.AbsoluteTolerance(dlarray(single(1e-5)));
            test.verifyThat(y(:, :, 1), iIsEqualTo(y1,'Within',tol));
            test.verifyThat(y(:, :, 2), iIsEqualTo(y2,'Within',tol));
        end
        
        function checkDuplicateSentence(test)
            % Check that the results per observation dimension match since
            % the two input data observations are equal.
            
            txt = "nipson anomemata me monan opsin";
            
            mdl = bert();
            seq = mdl.Tokenizer.encode(txt);
            x1 = dlarray(seq{1});
            
            % Duplicate x1 in observation dimension
            x = cat(3, x1, x1);
            y = bert.model(x, mdl.Parameters);
            
            tol = matlab.unittest.constraints.AbsoluteTolerance(dlarray(single(1e-5)));
            test.verifyThat(y(:, :, 1), iIsEqualTo(y(:, :, 2),'Within',tol));
        end
        
        function defaultOutputsIsLastLayer(test)
            % Verify the Outputs NVP default is the last layer.
            mdl = bert();
            x = dlarray(1:10);
            z1 = bert.model(x,mdl.Parameters);
            z2 = bert.model(x,mdl.Parameters,'Outputs',mdl.Parameters.Hyperparameters.NumLayers);
            test.verifyEqual(z1,z2);
        end
        
        function outputsCanBeUsed(test)
            % Verify the Outputs NVP can be used.
            mdl = bert();
            seqlen = 10;
            batchsize = 3;
            x = dlarray(randi([1,100],[1,seqlen,batchsize]));
            outputs = [1,3,5];
            [z1,z3,z5] = bert.model(x,mdl.Parameters,'Outputs',outputs);
            expSize = [mdl.Parameters.Hyperparameters.HiddenSize, seqlen, batchsize];
            test.verifySize(z1,expSize);
            test.verifySize(z3,expSize);
            test.verifySize(z5,expSize);
            % A final neat check is to modify the NumLayers and check the
            % value of z5
            mdl.Parameters.Hyperparameters.NumLayers = 5;
            z5_exp = bert.model(x,mdl.Parameters);
            test.verifyEqual(z5,z5_exp);
        end
        
        function outputsCanDuplicateAndBeOutOfOrder(test)
            % Subtle edge case
            mdl = bert();
            x = dlarray(1:10);
            outputs = [5,12,2,5,1,2];
            [z1,z2,z3,z4,~,z6] = bert.model(x,mdl.Parameters,'Outputs',outputs);
            test.verifyEqual(z1,z4);
            test.verifyEqual(z3,z6);
            z_exp = bert.model(x,mdl.Parameters);
            test.verifyEqual(z2,z_exp);
        end
        
        function negativeTestOutputs(test,InvalidOutputs)
            % Check the Outputs NVP errors in expected scenarios
            mdl = bert();
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,mdl.Parameters,'Outputs',InvalidOutputs.Value), InvalidOutputs.ErrorID);
        end
        
        function negativeTestSeparatorCode(test, InvalidCodes)
            % Check the SeparatorCode NVP errors in expected scenarios
            mdl = bert();
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,mdl.Parameters,'SeparatorCode',InvalidCodes.Value), InvalidCodes.ErrorID);
        end
        
        function negativeTestPaddingCode(test, InvalidCodes)
            % Check the PaddingCode NVP errors in expected scenarios
            mdl = bert();
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,mdl.Parameters,'PaddingCode',InvalidCodes.Value), InvalidCodes.ErrorID);
        end
        
        function negativeTestDropoutProb(test, InvalidDropoutProb)
            % Check the DropoutProb NVP errors in expected scenarios
            mdl = bert();
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,mdl.Parameters,'DropoutProb',InvalidDropoutProb.Value), InvalidDropoutProb.ErrorID);
        end
        
        function negativeTestAttentionDropoutProb(test, InvalidDropoutProb)
            % Check the AttentionDropoutProb NVP errors in expected scenarios
            mdl = bert();
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,mdl.Parameters,'AttentionDropoutProb',InvalidDropoutProb.Value), InvalidDropoutProb.ErrorID);
        end
        
        function negativeInputMask(test, InvalidInputMask)
            % Check the InputMask NVP errors in expected scenarios
            mdl = bert();
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,mdl.Parameters,'InputMask',InvalidInputMask.Value), InvalidInputMask.ErrorID);
        end
        
        function negativeInputX(test, InvalidInputX)
            % Check the input X errors in expected scenarios
            mdl = bert();
            test.verifyError(@() bert.model(InvalidInputX.Value,mdl.Parameters), InvalidInputX.ErrorID);
        end
        
        function negativeModelParameters(test, InvalidModelParams)
            % Check the input model paramters errors in expected scenarios
            x = dlarray([1,2,3]);
            test.verifyError(@() bert.model(x,InvalidModelParams.Value), InvalidModelParams.ErrorID);
        end
        
        function errorsForIncorrectModelName(test, InvalidModelName)
            % Check that 'Model' NVP errors in expected scenarios
            test.verifyError(@() bert('Model', InvalidModelName.Value), InvalidModelName.ErrorID);
        end
        
        function canUseInputMask(test)
            % Verify the InputMask NVP works as expected
            mdl = bert();
            strs = ["foo bar baz";"another exciting string with many words"];
            seqs = mdl.Tokenizer.encode(strs);
            [x,xmask] = padsequences(seqs,2,'PaddingValue',mdl.Tokenizer.PaddingCode);
            x = dlarray(x);
            % Specifying the mask should match the default when the mask
            % corresponds to padding tokens only.
            y_default = bert.model(x,mdl.Parameters);
            y_inputmask = bert.model(x,mdl.Parameters,'InputMask',xmask);
            test.verifyEqual(y_default,y_inputmask);
            % However the InputMask can be any logical with same size as x,
            % e.g. you may want to attend to padding for some reason.
            nomask = true(size(x));
            y_nomask = bert.model(x,mdl.Parameters,'InputMask',nomask);
            test.verifyNotEqual(y_inputmask,y_nomask);
        end
    end
end

function ids = fake_input_ids()
% Who was Jim Henson ? ||| Jim Henson was a puppeteer
ids = [101,2040,2001,3958,27227,1029,1064,1064,1064,3958,27227,2001,1037,13997,11510,102];
% matlab indexes from 1
ids = ids+1;
end

function constraint = iIsEqualTo(varargin)
constraint = matlab.unittest.constraints.IsEqualTo(varargin{:});
end

function constraint = iAbsoluteTolerance(tol)
constraint = matlab.unittest.constraints.AbsoluteTolerance(tol);
end

function s = iInvalidOutputs()
s = struct(...
    'NonInteger',iInvalidInputCase(1.1,'MATLAB:validators:mustBeInteger'),...
    'NonPositive', iInvalidInputCase(0,'MATLAB:validators:mustBePositive'),...
    'NonDouble', iInvalidInputCase("foo",'MATLAB:validators:mustBeNumericOrLogical'),...
    'TooBig', iInvalidInputCase([1,2,25],'MATLAB:validators:mustBeLessThanOrEqual'));
end

function s = iInvalidCodes()
s = struct(...
    'NonInteger', iInvalidInputCase(1.1,'MATLAB:validators:mustBeInteger'),...
    'NonPositive', iInvalidInputCase(0,'MATLAB:validators:mustBePositive'),...
    'NonNumeric', iInvalidInputCase("foo",'MATLAB:validators:mustBeNumericOrLogical'),...
    'InvalidDims', iInvalidInputCase([1,2,25],'MATLAB:validation:IncompatibleSize'),...
    'NonReal', iInvalidInputCase(1i,'MATLAB:validators:mustBeReal'));
end

function s = iInvalidDropoutProb()
s = struct(...
    'NonPositive', iInvalidInputCase(-1,'MATLAB:validators:mustBeNonnegative'), ...
    'Larger', iInvalidInputCase(1.5,'MATLAB:validators:mustBeLessThanOrEqual'), ...
    'InvalidDims', iInvalidInputCase([1,2],'MATLAB:validation:IncompatibleSize'), ...
    'NonReal', iInvalidInputCase(1i,'MATLAB:validators:mustBeReal'), ...
    'NonNumeric', iInvalidInputCase("foo",'MATLAB:validators:mustBeNumericOrLogical'));
end

function s = iInvalidInputMask()
s = struct(...
    'InvalidDims', iInvalidInputCase([true; true; false],"bert:model:InvalidMaskSize"), ...
    'InvalidTypeString', iInvalidInputCase("foo",'MATLAB:validators:mustBeNumericOrLogical'), ...
    'InvalidTypeCell', iInvalidInputCase({{1, 1, 0}},'MATLAB:validators:mustBeNumericOrLogical'));
end

function s = iInvalidInputX()
s = struct(...
    'InvalidTypeString', iInvalidInputCase("foo",'MATLAB:validation:UnableToConvert'), ...
    'InvalidTypeCell', iInvalidInputCase({{1}},'MATLAB:validation:UnableToConvert'), ...
    'Logical', iInvalidInputCase(true,'MATLAB:validators:mustBeNumeric'), ...
    'Complex', iInvalidInputCase(1 + 1i,'MATLAB:validation:UnableToConvert'), ...
    'IsEmpty', iInvalidInputCase([], 'MATLAB:validators:mustBeNonempty'));
end

function s = iInvalidModelParams()
s = struct(...
    'NonStructNumerical', iInvalidInputCase(10,'MATLAB:validators:mustBeA'), ...
    'NonStructString', iInvalidInputCase("foo",'MATLAB:validators:mustBeA'), ...
    'NonStructCell', iInvalidInputCase({10},'MATLAB:validators:mustBeA'), ...
    'IsEmpty', iInvalidInputCase([], 'MATLAB:validators:mustBeA'));
end

function s = iInvalidModelName()
s = struct(...
    'NonStructNumerical', iInvalidInputCase(10,'MATLAB:validators:mustBeMember'), ...
    'NonStructString', iInvalidInputCase("foo",'MATLAB:validators:mustBeMember'), ...
    'NonStructCell', iInvalidInputCase({10},'MATLAB:validators:mustBeMember'));
end

function s = iInvalidInputCase(val,id)
s = struct(...
    'Value',val,...
    'ErrorID',id);
end