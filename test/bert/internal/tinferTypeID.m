classdef tinferTypeID < matlab.unittest.TestCase
    properties(Constant)
        inferTypeID = @bert.internal.inferTypeID
    end
    
    methods(Test)
        function oneObsOneSeparator(test)
            % When there is one separator and one observation the expected
            % types are only 1.
            sep = 1;
            x = [2,3,sep];
            act = test.inferTypeID(x,sep);
            test.verifyEqual(act,ones(size(x)));
        end
        
        function oneObsOneSeparatorWithPadding(test)
            % When there is one separator followed by more data, the
            % expectation is that this data should be padding, and gets
            % type 1.
            sep = 1;
            x = [2,3,sep,4];
            act = test.inferTypeID(x,sep);
            test.verifyEqual(act,ones(size(x)));
        end
        
        function oneObsTwoSeparators(test)
            % When there are two separators the x should correspond to a
            % sentence-pair task. Label everything after the first separtor
            % as type 2.
            sep = 1;
            x = [2,3,sep,4,sep];
            exp = [1,1,1,2,2];
            act = test.inferTypeID(x,sep);
            test.verifyEqual(act,exp);            
        end
        
        function oneObsTwoSeparatorsEdgeCase(test)
            % Verify behaviour when there is no token between the two
            % separators
            sep = 1;
            x = [2,3,sep,sep];
            exp = [1,1,1,2];
            act = test.inferTypeID(x,sep);
            test.verifyEqual(act,exp);
        end
        
        function oneObsTwoSeparatorsPadded(test)
            % When tokens follow the second separator, mark them as type 1.
            % In proper workflows these should only ever be padding.
            sep = 1;
            x = [2,3,sep,4,sep,5,6];
            exp = [1,1,1,2,2,1,1];
            act = test.inferTypeID(x,sep);
            test.verifyEqual(act,exp);
        end
        
        function batchedCase(test)
            % Verify behaviour for a batched case
            sep = 1;
            pad = 100;
            x1 = [2,3,sep,pad,pad,pad,pad,pad];
            type1 = ones(size(x1));
            x2 = [4,5,6,sep,7,8,sep,pad];
            type2 = [1,1,1,1,2,2,2,1];
            x = cat(3,x1,x2);
            exp = cat(3,type1,type2);
            act = test.inferTypeID(x,sep);
            test.verifyEqual(act,exp)
        end
    end
end