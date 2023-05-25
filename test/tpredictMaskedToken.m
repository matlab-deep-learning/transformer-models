classdef(SharedTestFixtures={
        DownloadBERTFixture, DownloadJPBERTFixture}) tpredictMaskedToken < matlab.unittest.TestCase
    % tpredictMaskedToken   Unit test for predictMaskedToken
    
    % Copyright 2023 The MathWorks, Inc.

    properties(TestParameter)
        Models = {"tiny","japanese-base-wwm"}
        ValidText = iGetValidText;
    end
    
    methods(Test)
        function verifyOutputDimSizes(test, Models, ValidText)
            inSize = size(ValidText);
            mdl = bert("Model", Models);
            outputText = predictMaskedToken(mdl,ValidText);
            test.verifyEqual(size(outputText), inSize);
        end
        
        function maskTokenIsRemoved(test, Models)
            text = "This has a [MASK] token.";
            mdl = bert("Model", Models);
            outputText = predictMaskedToken(mdl,text);
            test.verifyFalse(contains(outputText, "[MASK]"));
        end

        function inputWithoutMASKRemainsTheSame(test, Models)
            text = "This has a no mask token.";
            mdl = bert("Model", Models);
            outputText = predictMaskedToken(mdl,text);
            test.verifyEqual(text, outputText);
        end
    end
end

function validText = iGetValidText
manyStrs = ["Accelerating the pace of [MASK] and science";
            "The cat [MASK] soundly.";
            "The [MASK] set beautifully."];
singleStr = "Artificial intelligence continues to shape the future of industries," + ...
    " as innovative applications emerge in fields such as healthcare, transportation," + ...
    " entertainment, and finance, driving productivity and enhancing human capabilities.";
validText = struct('StringsAsColumns',manyStrs,...
             'StringsAsRows',manyStrs',...
             'ManyStrings',repmat(singleStr,3),...
             'SingleString',singleStr,...
             'FirstWordMasked', "[MASK] the pace.",...
             'LastWordMasked', "Accelerating the [MASK].",...
             'SingleMaskToken', "[MASK]");
end