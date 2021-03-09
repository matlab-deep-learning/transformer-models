classdef tBasicTokenizer < matlab.unittest.TestCase
    % tBasicTokenizer   Unit tests for the BasicTokenizer
    
    % Copyright 2021 The MathWorks, Inc.    
    
    methods(Test)
        function canConstruct(test)
            tok = bert.tokenizer.internal.BasicTokenizer();
            test.verifyClass(tok,'bert.tokenizer.internal.BasicTokenizer');
            test.verifyInstanceOf(tok,'bert.tokenizer.internal.Tokenizer');
        end
        
        function canTokenize(test)
            tok = bert.tokenizer.internal.BasicTokenizer();
            str = "foo bar baz";
            exp_out = ["foo","bar","baz"];
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);           
        end
        
        function removesControlCharactersAndWhitespace(test)
            tok = bert.tokenizer.internal.BasicTokenizer();
            aControlChar = compose('\x000A');
            aFormatChar = compose('\xDB40\xDC7E');
            aSpaceChar = compose('\x00A0');
            words = ["Testing","a","blah"];
            str = strcat(words(1)," ",aFormatChar," ",...
                words(2),aFormatChar," ",aControlChar," ",words(3),aSpaceChar);
            exp_out = [lower(words(1)),words(2),words(3)];
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);
        end
        
        function splitsOnNewlines(test)
            % Regression test for a bug
            tok = bert.tokenizer.internal.BasicTokenizer();
            str = "hello"+newline+"world";
            act_toks = tok.tokenize(str);
            exp_toks = ["hello","world"];
            test.verifyEqual(act_toks,exp_toks);
        end
        
        function tokenizesCJK(test)
            tok = bert.tokenizer.internal.BasicTokenizer();            
            str = strcat(...
                compose("Arbitrary \x4E01\x4E02 CJK chars \xD869\xDF00\xD86D\xDE3E"),...
                "more");
            exp_out = ["arbitrary",compose("\x4E01"),compose("\x4E02"),"cjk","chars",...
                compose("\xD869\xDF00"),compose("\xD86D\xDE3E"),"more"];
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);
        end
        
        function splitsOnPunctuation(test)
            tok = bert.tokenizer.internal.BasicTokenizer(); 
            str = "hello. hello, world? hello world! hello";
            exp_out = ["hello",".","hello",",","world","?","hello","world","!","hello"];
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);
        end
        
        function stripsAccents(test)
            tok = bert.tokenizer.internal.BasicTokenizer();
            str = compose("h\x00E9llo");
            exp_out = "hello";
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);
        end
        
        function canBeCaseSensitive(test)
            tok = bert.tokenizer.internal.BasicTokenizer('IgnoreCase',false);
            str = "FOO bAr baz";
            exp_out = ["FOO","bAr","baz"];
            act_out = tok.tokenize(str);
            test.verifyEqual(act_out,exp_out);
        end
    end
end