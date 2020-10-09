function path = getRepoRoot()
% getRepoRoot   Return a path to the repository's root directory.

% Copyright 2020 The MathWorks, Inc.

thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);

% the root is up two directories (<root>/test/tools/getRepoRoot.m)
path = fullfile(thisDir,'..','..');
end