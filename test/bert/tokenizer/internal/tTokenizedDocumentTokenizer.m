classdef tTokenizedDocumentTokenizer < matlab.unittest.TestCase
    % tTokenizedDocumentTokenizer   Unit tests for TokenizedDocumentTokenizer.
    
    % Copyright 2023 The MathWorks, Inc.
    
    methods(Test)
        function tokenizationMatchesTokenizedDocument(test)
            % TokenizedDocumentTokenizer does what it says on the tin -
            % uses tokenizedDocument.
            tok = bert.tokenizer.internal.TokenizedDocumentTokenizer;
            str = "a random string. doesn't matter.";
            toks = tok.tokenize(str);
            doc = tokenizedDocument(str);
            toksExp = {string(doc)};
            test.verifyEqual(toks,toksExp);
        end

        function canSetOptions(test)
            % We can pass in tokenization options matching
            % tokenizedDocument's NVPs.
            customToken = "foo bar";
            tok = bert.tokenizer.internal.TokenizedDocumentTokenizer(CustomTokens=customToken);
            str = "in this case "+customToken+" is one token.";
            toks = tok.tokenize(str);
            import matlab.unittest.constraints.AnyElementOf
            import matlab.unittest.constraints.IsEqualTo
            test.verifyThat(AnyElementOf(toks{1}),IsEqualTo(customToken));
        end
    end
end