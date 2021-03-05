classdef DownloadGPT2Fixture < matlab.unittest.fixtures.Fixture
    % DownloadGPT2Fixture   A fixture for calling gpt2.download if
    % necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.
    
    % Copyright 2020 The MathWorks, Inc
    
    properties(Constant)
        GPT2DataDir = fullfile(matlab.internal.examples.utils.getSupportFileDir(),"nnet","data","networks")
        GPT2DataFiles = ["gpt2_355M_params.mat","gpt2_encoder.txt","gpt2_vocab.bpe"]
    end
    
    properties
        DataDirExists (1,1) logical
        DataFileExists
    end
    
    methods
        function setup(this)
            this.DataDirExists = exist(this.GPT2DataDir,'dir')==7;
            dataFileExists = arrayfun(@(file) exist(fullfile(this.GPT2DataDir,file),'file')==2, this.GPT2DataFiles);
            this.DataFileExists = containers.Map(this.GPT2DataFiles,dataFileExists);
            if ~this.DataDirExists || any(~dataFileExists)
                % Call this in eval to capture and drop any standard output
                % that we don't want polluting the test logs.
                evalc('gpt2();');
            end
        end
        
        function teardown(this)
            if ~this.DataDirExists
                rmdir(this.GPT2DataDir,'s');
            end
            for file = this.GPT2DataFiles
                if ~this.DataFileExists(file)
                    delete(fullfile(this.GPT2DataDir,file));
                end
            end
        end
    end
end