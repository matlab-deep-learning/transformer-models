function path = convertModelNameToDirectories(name)
% convertModelNameToDirectories   Converts the user facing model name to
% the directory name used by support files.

% Copyright 2021 The MathWorks, Inc.
arguments
    name (1,1) string
end
path = {"data","networks","finbert",name};
end