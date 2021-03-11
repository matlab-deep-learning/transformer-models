function filePath = getSupportFilePath(fileName)
% getSupportFilePath   This function is for converting any differences
% between the model names presented to the user and the support files
% URLs.

% Copyright 2021 The MathWorks, Inc.
arguments
    fileName (1,1) string
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