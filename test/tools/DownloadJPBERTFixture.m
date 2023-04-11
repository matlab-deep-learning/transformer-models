classdef DownloadJPBERTFixture < matlab.unittest.fixtures.Fixture
    % DownloadJPBERTFixture   A fixture for downloading the Japanese BERT models and
    % clearing them out after tests finish if they were not previously
    % downloaded.
    
    % Copyright 2023 The MathWorks, Inc
    
    properties(Constant)
        Models = dictionary(["japanese-base", "japanese-base-wwm"], ...
            ["bert-base-japanese", "bert-base-japanese-whole-word-masking"]);
    end
    
    properties
        DataDirExists
    end    
    
    methods        
        function setup(this)
            dirs = this.pathToSupportFile(this.Models.values);
            dataDirsExist = arrayfun(@(dir) exist(dir,'dir')==7, dirs);
            this.DataDirExists = dictionary(this.Models.keys,dataDirsExist);
            modelNames = this.Models.keys;
            for i=1:numel(modelNames)
                model = modelNames(i);
                if ~this.DataDirExists(model)
                    bert('Model',model);
                end
            end
        end
        
        function teardown(this)
            modelNames = this.Models.keys;
            for i=1:numel(modelNames)
                model = modelNames(i);
                if ~this.DataDirExists(model)
                    rmdir(this.pathToSupportFile(this.Models(model)),'s');
                end
            end
        end
    end    
    
    methods(Access=private)
        function path = pathToSupportFile(~,model)
            modelDir = fullfile("data", "networks", "ja_bert", model);
            path = fullfile(matlab.internal.examples.utils.getSupportFileDir(),"nnet",modelDir);
        end
    end
end