classdef downloadGPT2Fixture < matlab.unittest.fixtures.Fixture
    % downloadGPT2Fixture   A fixture for calling gpt2.download if
    % necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.
    
    % Copyright 2020 The MathWorks, Inc
    
    properties(Constant)
        GPT2DataDir = fullfile(fileparts(mfilename('fullpath')),'..','..','gpt2-355M')
    end
    
    properties
        DataDirExists (1,1) logical
    end
    
    methods
        function setup(this)
            this.DataDirExists = exist(this.GPT2DataDir,'dir')==7;
            if ~this.DataDirExists                
                gpt2.download();
            end
        end
        
        function teardown(this)
            if ~this.DataDirExists
                rmdir(this.GPT2DataDir,'s');
            end
        end
    end
end