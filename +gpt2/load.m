function newParameters = load(filepath)
% load   Load GPT-2
%
%   parameters = load(filepath) will load a GPT-2 model from the directory
%   specified by filepath.

% Load the parameters
s = load(filepath);

% First, assign the hyperparameters and then remove them from the loaded
% structure.
newParameters = struct;
newParameters.Hyperparameters = s.hyperparameters;
s = rmfield(s, 'hyperparameters');

% Next, assign the weights
newParameters.Weights = struct;

originalWeightNames = fieldnames(s);
newWeightNames = erase(originalWeightNames, 'model_');

% Permute all of the weights to match the MATLAB format
for i = 1:numel(originalWeightNames)
    if startsWith(newWeightNames{i},'h')
        nameSplit = split(newWeightNames{i}, '_');
        newParameters.Weights.(char(nameSplit(1))).(char(join(nameSplit(2:end),'_'))) = ...
            s.(originalWeightNames{i});
    else
        newParameters.Weights.(char(newWeightNames{i})) = s.(originalWeightNames{i});
    end
end

newParameters.Weights = iStructRecurseFun(@dlarray, newParameters.Weights);

end

function s = iStructRecurseFun(F, s, varargin)
if ~isstruct(s)
    s = F(s, varargin{:});
else
    fn = fieldnames(s);
    for i = 1:numel(fn)
        others = cellfun(@(o)o.(fn{i}), varargin, 'UniformOutput', false);
        s.(fn{i}) = iStructRecurseFun(F, s.(fn{i}), others{:});
    end
end
end