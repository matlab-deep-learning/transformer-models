classdef(SharedTestFixtures = {
        DownloadBERTFixture}) tWordPieceTokenizer < matlab.unittest.TestCase
    % tWordPieceTokenizer   Unit tests for WordPieceTokenizer
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(TestParameter)
        ModelAndExpectedVocabSize = struct(...
            'uncased',struct('Model','base','VocabSize',30522),...
            'multilingualCased',struct('Model','multilingual-cased','VocabSize',119547))
    end
    
    methods(Test)
        function canConstruct(test)
            enc = wordEncoding("foo");
            tok = bert.tokenizer.internal.WordPieceTokenizer(enc);
            test.verifyClass(tok,'bert.tokenizer.internal.WordPieceTokenizer');
            test.verifyInstanceOf(tok,'bert.tokenizer.internal.Tokenizer');
        end
        
        function canConstructWithFile(test)
            words = ["foo","bar"];
            enc = wordEncoding(words);
            fixture = matlab.unittest.fixtures.TemporaryFolderFixture();
            test.applyFixture(fixture);
            folder = fixture.Folder;
            testVocab = fullfile(folder,"testvocab.txt");
            fid = fopen(testVocab,'w','n','utf-8');
            fprintf(fid,join(words,newline));
            fclose(fid);
            tokFromEncoding = bert.tokenizer.internal.WordPieceTokenizer(enc);
            tokFromFile = bert.tokenizer.internal.WordPieceTokenizer(testVocab);
            test.verifyEqual(tokFromEncoding.Vocab,tokFromFile.Vocab);
        end
        
        function canSetUnknownToken(test)
            enc = wordEncoding("foo");
            unk = "bar";
            tok = bert.tokenizer.internal.WordPieceTokenizer(enc,'UnknownToken',unk);
            test.verifyEqual(tok.Unk,unk)
            str = "blah";
            ustr = textanalytics.unicode.UTF32(str);
            act_out = tok.tokenize(ustr);
            exp_out = unk;
            test.verifyEqual(act_out,exp_out);
        end
        
        function canSetMaxTokenLength(test)
            enc = wordEncoding("foo");
            maxLen = 2;
            tok = bert.tokenizer.internal.WordPieceTokenizer(enc,'MaxTokenLength',maxLen);
            test.verifyEqual(tok.MaxChar,maxLen);
            str = "foo";
            ustr = textanalytics.unicode.UTF32(str);
            act_out = tok.tokenize(ustr);
            exp_out = tok.Unk;
            test.verifyEqual(act_out,exp_out);
        end
        
        function canTokenize(test)
            enc = wordEncoding(["foo","bar","##foo"]);
            tok = bert.tokenizer.internal.WordPieceTokenizer(enc);
            str = "foo bar foobar barba bafoobar barfoo";
            wsTok = bert.tokenizer.internal.WhitespaceTokenizer;
            ustr = textanalytics.unicode.UTF32(wsTok.tokenize(str));
            act_out = tok.tokenize(ustr);
            exp_out = ["foo","bar",tok.Unk,tok.Unk,tok.Unk,"bar","##foo"];
            test.verifyEqual(act_out,exp_out);
        end
        
        function hasExpectedVocabSize(test,ModelAndExpectedVocabSize)
            vocab = bert.internal.getSupportFilePath(ModelAndExpectedVocabSize.Model,"vocab.txt");
            tok = bert.tokenizer.internal.WordPieceTokenizer(vocab);
            test.verifyEqual(tok.Vocab.NumWords,ModelAndExpectedVocabSize.VocabSize);
        end
    end
end