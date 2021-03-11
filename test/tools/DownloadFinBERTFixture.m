classdef DownloadFinBERTFixture < matlab.unittest.fixtures.Fixture
    % DownloadFinBERTFixture   A fixture for calling finbert.download if
    % necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.
    
    % Copyright 2021 The MathWorks, Inc
    
    properties(Constant)
        Models = ["sentiment-model","language-model"]
    end
    
    properties
        DataDirExists
    end
    
    methods
        function setup(this)
            dirs = this.pathToSupportFile(this.Models);
            dataDirsExist = arrayfun(@(dir) exist(dir,'dir')==7, dirs);
            this.DataDirExists = containers.Map(this.Models,dataDirsExist);
            for i=1:numel(this.Models)
                model = this.Models(i);
                if ~this.DataDirExists(model)
                    finbert('Model',model);
                end
            end
        end
        
        function teardown(this)
            for i=1:numel(this.Models)
                model = this.Models(i);
                if ~this.DataDirExists(model)
                    rmdir(this.pathToSupportFile(model),'s');
                end
            end
        end
    end
    
    methods(Access=private)
        function path = pathToSupportFile(this,model)
            modelDir = this.convertModelNameToDirectories(model);
            path = fullfile(matlab.internal.examples.utils.getSupportFileDir(),"nnet",modelDir);
        end
        
        function modelDirs = convertModelNameToDirectories(~,modelNames)
            modelDirs = arrayfun(@(model) finbert.internal.convertModelNameToDirectories(model), modelNames, 'UniformOutput', false);
            modelDirs = cellfun(@(dirCell) fullfile(dirCell{:}), modelDirs);
        end
    end
end