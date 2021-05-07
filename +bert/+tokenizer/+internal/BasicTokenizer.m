classdef BasicTokenizer < bert.tokenizer.internal.Tokenizer
    % BasicTokenizer   Perform basic tokenization.
    
    % Copyright 2020 The MathWorks, Inc.
    
    properties(SetAccess=private)
        IgnoreCase
    end
    
    properties(Constant,Access=private)
        WhiteSpaceTokenizer = bert.tokenizer.internal.WhitespaceTokenizer()
    end
    
    methods
        function this = BasicTokenizer(nvp)
            % BasicTokenizer()   Constructs a BasicTokenizer.
            %
            % Name-Value Pair Arguments:
            %   'IgnoreCase'   - When set to true the text to be tokenized
            %                    is lower cased during tokenization.
            %                    Default is true.
            arguments
                nvp.IgnoreCase (1,1) logical = true
            end
            this.IgnoreCase = nvp.IgnoreCase;
        end
        
        function tokens = tokenize(this,text)
            arguments
                this (1,1) bert.tokenizer.internal.BasicTokenizer
                text (1,1) string
            end
            u = textanalytics.unicode.UTF32(text);
            u = this.cleanText(u);
            u = this.tokenizeCJK(u);
            text = u.string();
            if this.IgnoreCase
                text = lower(text);
                text = textanalytics.unicode.nfd(text);
            end
            u = textanalytics.unicode.UTF32(text);
            cats = u.characterCategories('Granularity','detailed');
            if this.IgnoreCase
                [u,cats] = this.stripAccents(u,cats);
            end
            tokens = this.splitOnPunc(u,cats);
            tokens = join(cat(2,tokens{:})," ");
            tokens = this.whiteSpaceTokenize(tokens);
        end
    end
    
    methods(Access=private)
        function tok = whiteSpaceTokenize(this,text)
            % Simple whitespace tokenization
            tok = this.WhiteSpaceTokenizer.tokenize(text);
        end
        
        function u = cleanText(this,u)
            % normalize whitespace, remove control chars.
            udata = u.Data;
            ucats = u.characterCategories('Granularity','detailed');
            ucats = ucats{1};
            isWhitespace = this.whitespaceIdx(udata,ucats);
            isControl = this.controlIdx(udata,ucats);
            udata(isWhitespace) = uint32(' ');
            udata(isControl) = [];
            u.Data = udata;
        end
        
        function tf = whitespaceIdx(~,udata,cats)
            whitespaceChar = uint32(sprintf(' \n\r\t')).';
            whitespaceCat = 'Zs';
            tf = (any(udata==whitespaceChar,1)|any(string(cats)==whitespaceCat,1));
        end
        
        function tf = controlIdx(~,udata,cats)
            % We want to keep certain whitespace control characters
            controlExceptions = uint32(sprintf('\t\n\r')).';
            controlChar = uint32([0;0xfffd]);
            controlCat = ["Cc";"Cf"];
            isControlCat = ~any(udata==controlExceptions,1) & any(string(cats)==controlCat,1);
            tf = (any(udata==controlChar,1)|isControlCat);
        end
        
        function out = splitOnPunc(~,uarray,catsCell)
            % splits on punctuation characters
            out = cell(numel(uarray),1);
            function u = createUTF32FromData(data)
                u = textanalytics.unicode.UTF32();
                u.Data = data;
            end
            for i = 1:numel(uarray)
                u = uarray(i);
                cats = catsCell{i};
                isPunc = isPunctuation(u,cats);
                if ~any(isPunc)
                    % early return
                    out{i} = u.string();                    
                else
                    isBeforePunc = circshift(isPunc,-1);
                    isBeforePunc(end) = 0;
                    splitBoundary = isPunc | isBeforePunc;
                    % We always split at the end if it's not already a boundary
                    splitBoundary(end) = 1;
                    splitLens = diff([0,find(splitBoundary)]);
                    newdata = mat2cell(u.Data,1,splitLens);
                    newU = cellfun(@(data) createUTF32FromData(data), newdata);
                    out{i} = newU.string();
                end
            end
        end
        
        function u = tokenizeCJK(~,u)
            udata = u.Data;
            if isempty(udata)
                return
            end
            cjkCodepoints = isCJK(udata);
            % For each CJK codepoint we add a preceding and succeeding space
            newIdxDelta = 2*cumsum(cjkCodepoints);
            % But only the preceding space affects the position of the CJK
            % codepoint.
            newIdxDelta(cjkCodepoints)=newIdxDelta(cjkCodepoints)-1;
            newIdx = (1:numel(udata)) + newIdxDelta;
            newdata = uint32(' ').*ones(1,newIdx(end),'uint32');
            newdata(newIdx) = udata;
            u.Data = newdata;
        end
        
        function [u,cats] = stripAccents(~,u,cats)
            for i = 1:numel(u)
                isMn = cats{i}=="Mn";
                u(i).Data(isMn) = [];
                cats{i}(isMn) = [];
            end
        end
    end
end

function tf = inRange(num,lower,upper)
tf = num>=lower & num<=upper;
end

function tf = isCJK(udata)
tf = inRange(udata,0x4E00,0x9FFF)|...
    inRange(udata,0x3400,0x4DBF)|...
    inRange(udata,0x20000,0x2A6DF)|...
    inRange(udata,0x2A700,0x2B73F)|...
    inRange(udata,0x2B820,0x2CEAF)|...
    inRange(udata,0xF900,0xFAFF)|...
    inRange(udata,0x2F800,0x2FA1F);
end

function tf = isPunctuation(u,cats)
udata = u.Data;
tf = ...
    inRange(udata,33,47)|...
    inRange(udata,58,64)|...
    inRange(udata,91,96)|...
    inRange(udata,123,126);
cats = string(cats);
tf = (tf)|(cats.startsWith("P"));
end