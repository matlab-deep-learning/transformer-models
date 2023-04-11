classdef(SharedTestFixtures = {
        DownloadJPBERTFixture}) tBERTTokenizerForJP < matlab.unittest.TestCase
    % tBERTTokenizerForJP   Unit tests for the BERTTokenizer using Japanese
    % BERT models.
    
    % Copyright 2023 The MathWorks, Inc.
    
    properties(TestParameter)
        VocabFiles = iVocabFiles()
    end
    
    properties(Constant)
        Constructor = @iJapaneseTokenizerConstructor
    end
    
    methods(Test)
                
        function hasExpectedProperties(test, VocabFiles)
            tok = test.Constructor(VocabFiles);
            test.verifyEqual(tok.PaddingToken, "[PAD]");
            test.verifyEqual(tok.StartToken, "[CLS]");
            test.verifyEqual(tok.SeparatorToken, "[SEP]");
            test.verifyEqual(tok.PaddingCode, 1);
            test.verifyEqual(tok.SeparatorCode, 4);
            test.verifyEqual(tok.StartCode, 3);
        end
        
        function matchesExpectedEncoding(test, VocabFiles)
            tok = test.Constructor(VocabFiles);
            text = "月夜の想い、謎めく愛。君の謎。";
            expectedEncoding = [3 38 29340 6 12385 7 5939 2088 28504 768 ...
                9 2607 6 5939 9 4];
            y = tok.encode(text);
            test.verifyClass(y,'cell');
            y1 = y{1};
            test.verifyEqual(y1(1),tok.StartCode);
            test.verifyEqual(y1(end),tok.SeparatorCode);
            test.verifyEqual(y1,expectedEncoding);
        end    
    end    
end

function modelNames = iModelNames
% struct friendly model names
modelNames = ["japanese_base", "japanese_base_wwm"];
end

function vocabFiles = iVocabFiles
modelDir = ["bert-base-japanese", "bert-base-japanese-whole-word-masking"];
modelNames = iModelNames;
vocabFiles = struct();
for i = 1:numel(modelNames)
    versionName = modelDir(i);
    vocabDir = fullfile("data", "networks", "ja_bert", versionName, "vocab.txt");
    model = modelNames(i);
    vocabFiles.(replace(model, "-", "_")) = fullfile(matlab.internal.examples.utils.getSupportFileDir(),"nnet",vocabDir);
end
end

function japaneseBERTTokenizer = iJapaneseTokenizerConstructor(vocabLocation)
btok = bert.tokenizer.internal.TokenizedDocumentTokenizer("Language","ja","TokenizeMethod","mecab",IgnoreCase=false);
ftok = bert.tokenizer.internal.FullTokenizer(vocabLocation,BasicTokenizer=btok);
japaneseBERTTokenizer = bert.tokenizer.BERTTokenizer(vocabLocation,FullTokenizer=ftok);
end    