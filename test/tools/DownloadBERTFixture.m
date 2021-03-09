classdef DownloadBERTFixture < matlab.unittest.fixtures.Fixture
    % DownloadBERTFixture   A fixture for downloading the BERT models and
    % clearing them out after tests finish if they were not previously
    % downloaded.
    
    % Copyright 2021 The MathWorks, Inc
    
    properties(Constant)
        Models = [
            "base"
            "multilingual-cased"
            "tiny"]
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
                    bert('Model',model);
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
            modelDirs = arrayfun(@(model) bert.internal.convertModelNameToDirectories(model), modelNames, 'UniformOutput', false);
            modelDirs = cellfun(@(dirCell) fullfile(dirCell{:}), modelDirs);
        end
    end
end