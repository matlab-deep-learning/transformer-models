classdef(SharedTestFixtures = {DownloadGPT2Fixture}) tGPT2Tokenizer < matlab.unittest.TestCase
    % tGPT2Tokenizer   Tests for the text encoder class
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties
        % Setup a default encoder for testing
        Encoder
    end
    
    properties(TestParameter)
        % UnicodeTextAndEncodedText
        UnicodeTextAndEncodedText = iGetUnicodeTextAndEncodedText()
        
        % Strings to be tokenized by the regex and expected results
        StringToTokenize = iStringToTokenize()
        
        % Test strings for encoding
        StringToEncode = iStringToEncode()
    end
    
    methods(TestClassSetup)
        function setup(test)
            test.setupEncoder();
        end
    end
    
    methods(Test)
        function encodeGivesCorrectResults(test, UnicodeTextAndEncodedText)
            % Unpack the test parameter
            inputText = UnicodeTextAndEncodedText.UnicodeText;
            expectedOutputText = UnicodeTextAndEncodedText.EncodedText;
            
            % Create the encoder
            enc = gpt2.tokenizer.GPT2Tokenizer(iModelName(), iModelPath());
            
            % Get the output text
            actualOutputText = enc.encode(inputText);
            
            % Verify the output is correct
            test.verifyThat( ...
                actualOutputText, iIsEqualTo(expectedOutputText) );
        end
        
        function decodeGivesCorrectResults(test, UnicodeTextAndEncodedText)
            % Unpack the test parameter
            inputText = UnicodeTextAndEncodedText.EncodedText;
            expectedOutputText = UnicodeTextAndEncodedText.UnicodeText;
            
            % Create the encoder
            enc = gpt2.tokenizer.GPT2Tokenizer(iModelName(), iModelPath());
            
            % Get the output text
            actualOutputText = enc.decode(inputText);
            
            % Verify the output is correct
            test.verifyThat( ...
                actualOutputText, iMatches(expectedOutputText) );
        end        
        
        function testTokenization(test, StringToTokenize)
            % Test the regex tokenization used by the Encoder.
            enc = test.Encoder;
            toks = iTokenize(StringToTokenize.string,enc);
            test.verifyEqual(toks,StringToTokenize.exp);
        end        
        
        function byteEncoderSize(test)
            % Verify the expected size of the byte encoder.
            enc = test.Encoder;
            test.verifyEqual(numel(unique(enc.ByteEncoder)),256);
        end
        
        function independenceOfInputClass(test, StringToEncode)
            % Verify that it doesn't matter if input text is string or char
            enc = test.Encoder;
            str = convertCharsToStrings(StringToEncode);
            ch = convertStringsToChars(StringToEncode);
            test.verifyEqual(enc.encode(str),enc.encode(ch));
        end
        
        function commentNotInBPE(test)
            % The vocab.bpe includes a comment that we must manually strip
            % out. Ensure we do that.
            vocabFile = gpt2.internal.getSupportFilePath("gpt2_vocab.bpe");
            fid = fopen(vocabFile,'r','n','utf-8');
            s = textscan(fid,'%s', 'Delimiter', '\n');
            fclose(fid);
            s = s{1};
            % first line of vocab.bpe is the comment
            firstLine = s(1);
            % split as in Encoder and check we don't accidentally add these
            % values to BPERanks.
            splitfirstLine = split(convertCharsToStrings(firstLine));
            enc = test.Encoder;
            actBPE = enc.BPERanks;
            % splitfirstLine is a string, ensure actBPE is
            test.assertClass(actBPE,'string');
            actBPEMatchesComment = any(actBPE==splitfirstLine(1),'all') || any(actBPE==splitfirstLine(2),'all');
            test.verifyFalse(actBPEMatchesComment, "A match was found between the encoder's BPERanks and the comment in vocab.bpe");
        end
        
        function decodeInvertsEncode(test,StringToEncode)
            % Ensure decoding inverts encoding
            enc = test.Encoder;
            encoded = enc.encode(StringToEncode);
            decoded = enc.decode(encoded);
            test.verifyMatches(decoded,StringToEncode);
        end
    end
    
    methods(Access=private)
        function setupEncoder(test)
            % Setup an encoder to be used by multiple tests.
            % Note this is a handle.
            test.Encoder = gpt2.tokenizer.GPT2Tokenizer(iModelName(),iModelPath());
        end
    end
end

function modelName = iModelName()
modelName = 'gpt2-355M';
end

function modelPath = iModelPath()
modelPath = fullfile(getRepoRoot());
end

function constraint = iIsEqualTo(varargin)
constraint = matlab.unittest.constraints.IsEqualTo(varargin{:});
end

function constraint = iMatches(varargin)
constraint = matlab.unittest.constraints.Matches(varargin{:});
end

function parameter = iGetUnicodeTextAndEncodedText()

parameter = struct;

parameter.NormalSentenceFragment1 = struct( ...
    'UnicodeText', 'In this tutorial we will see', ...
    'EncodedText', [818 428 11808 356 481 766] + 1 );

parameter.NonsenseFragment1 = struct( ...
    'UnicodeText', 'esgarghr', ...
    'EncodedText', [274 4563 456 81] + 1 );

% Weird edge case I found when I wasn't doing the Unicode decoding
% correctly. If you set "k=1" and input the sentence "In this tutorial ",
% into GPT-2 the first output is a special Unicode character known as a
% "Non-breaking space". After thinking about this, this weird behaviour is
% likely due to the fact that the tokenizer for GPT-2 tokenizes words so
% that they are preceded by spaces.
parameter.NonBreakingSpace = struct( ...
    'UnicodeText', native2unicode([194 160], 'UTF-8'), ...
    'EncodedText', 1849 + 1 );

end

function c = iStringToEncode()
% A test parameter of strings to be passed to the encoder.
c = {
    'In this tutorial we will see'
    'esgarghr'
    'foo bar baz'
    'A word and numbers 123 or 456word and decimal 7.89'
    'particles like don''t do that'
    'multiple white    space characters'
    };
end

function s = iStringToTokenize()
% A test parameter of strings to be tokenized using the regular expression
% attached to tokenizer.GPT2Tokenizer, and the expected output tokens.
s = struct( ...
    'oneWord', iStringToTokenizeStruct("foo","foo"), ...
    'twoWords', iStringToTokenizeStruct("foo bar",["foo"," bar"]), ...
    'possessive', iStringToTokenizeStruct("foo's",["foo","'s"]), ...
    'not', iStringToTokenizeStruct("foo't", ["foo","'t"]), ...
    'are', iStringToTokenizeStruct("foo're", ["foo","'re"]), ...
    'have', iStringToTokenizeStruct("foo've", ["foo","'ve"]), ...
    'm', iStringToTokenizeStruct("foo'm",["foo","'m"]), ...
    'will', iStringToTokenizeStruct("foo'll",["foo","'ll"]), ...
    'would', iStringToTokenizeStruct("foo'd",["foo","'d"]), ...
    'numbers', iStringToTokenizeStruct("foo 123bar",["foo"," 123","bar"]), ...
    'multiWS', iStringToTokenizeStruct("foo   bar",["foo", "  ", " bar"]));
end

function s = iStringToTokenizeStruct(str,expTokens)
% A struct for a case of the StringToTokenize test parameter.
s = struct('string',str,'exp',expTokens);
end

function toks = iTokenize(str,enc)
% Tokenize equivalently to tokenizer.GPT2Tokenizer
[toks,~] = regexp(str,enc.TokenizationExpression,'match','split');
end