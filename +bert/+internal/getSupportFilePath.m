function filePath = getSupportFilePath(modelName,fileName)
% getSupportFilePath   This function is for converting any differences
% between the model names presented to the user and the support files
% URLs.

% Copyright 2021 The MathWorks, Inc.
arguments
    modelName (1,1) string
    fileName (1,1) string
end
directory = bert.internal.convertModelNameToDirectories(modelName);
sd = matlab.internal.examples.utils.getSupportFileDir();
localFile = fullfile(sd,"nnet",directory{:},fileName);
if exist(localFile,'file')~=2
    disp("Downloading "+fileName+" to: "+localFile);
end
fileURL = strjoin([directory,fileName],"/");
filePath = matlab.internal.examples.downloadSupportFile("nnet",fileURL);
end