function filePath = getSupportFilePath(fileName)
% getSupportFilePath   This function is for converting any differences
% between the model names presented to the user and the support files
% URLs.

% Copyright 2021 The MathWorks, Inc.
arguments
    fileName (1,1) string
end
if ismember(version('-release'),["2020a","2020b"])
    filePath = legacySupportFilePath(fileName);
    return
end
sd = matlab.internal.examples.utils.getSupportFileDir();
localFileDir = {"data","networks"};%#ok
localFile = fullfile(sd,"nnet",localFileDir{:},fileName);
if exist(localFile,'file')~=2
    disp("Downloading "+fileName+" to: "+localFile);
end
fileURL = strjoin([localFileDir,fileName],"/");
filePath = matlab.internal.examples.downloadSupportFile("nnet",fileURL);
end

function filePath = legacySupportFilePath(fileName)
% For releases before matlab.internal.examples.downloadSupportFile,
% use manual download code. We save to the repo's root directory instead of
% the userpath.
% Create directories for the model.
modelType = 'gpt2-355M';
modelDirectory = fullfile(fileparts(mfilename('fullpath')),'..','..',modelType);
filePath = fullfile(modelDirectory,fileName);
iCreateDirectoryIfItDoesNotExist(modelDirectory);

iDownloadFileIfItDoesNotExist( ...
    filePath, ...
    "https://ssd.mathworks.com/supportfiles/nnet/data/networks/"+fileName );
end

function iCreateDirectoryIfItDoesNotExist(directory)
if ~exist(directory, 'dir')
    fprintf('Creating directory ''%s''...\n', directory);
    mkdir(directory);
else
    fprintf('Skipped creating directory ''%s'' as it already exists\n', directory);
end
end

function iDownloadFileIfItDoesNotExist(destination, source)
if ~exist(destination, 'file')
    fprintf('Downloading file ''%s'' ...\n', destination);
    websave(destination, source);
else
    fprintf('Skipped downloading file ''%s'' as it already exists\n', destination);
end
end