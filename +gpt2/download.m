function download()
% download   Download all of the necessary weight files for transformer model
%
%   download() will download all of the files that define the pretrained
%   GPT-2 355M model.

% Create directories for the model.
modelType = 'gpt2-355M';
modelDirectory = fullfile(fileparts(mfilename('fullpath')),'..',modelType);
iCreateDirectoryIfItDoesNotExist(modelDirectory);

% Download 'encoder.txt'. This is equivalent to 'encoder.json' from the
% original GPT-2.
iDownloadFileIfItDoesNotExist( ...
    fullfile(modelDirectory,'encoder.txt'), ...
    'https://ssd.mathworks.com/supportfiles/nnet/data/networks/gpt2_encoder.txt' );

% Download 'vocab.bpe'. This file contains the BPE ranks for the encoder.
% This file is identical to the one used by the original OpenAI repo.
iDownloadFileIfItDoesNotExist( ...
    fullfile(modelDirectory,'vocab.bpe'), ...
    'https://ssd.mathworks.com/supportfiles/nnet/data/networks/gpt2_vocab.bpe' );

% Download 'parameters.mat'. This contains all of the weights in the GPT-2
% model. They have been exported from the original TensorFlow
% implementation.
iDownloadFileIfItDoesNotExist( ...
    fullfile(modelDirectory,'parameters.mat'), ...
    'https://ssd.mathworks.com/supportfiles/nnet/data/networks/gpt2_355M_params.mat' );
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