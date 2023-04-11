function dirpath = convertModelNameToDirectories(name)
% convertModelNameToDirectories   Converts the user facing model name to
% the directory name used by support files.

% Copyright 2021-2023 The MathWorks, Inc.
arguments
    name (1,1) string
end
modelName = userInputToSupportFileName(name);
bertBaseLocation = "bert";
if contains(name,"japanese")
    bertBaseLocation = "ja_" + bertBaseLocation;
end
dirpath = {"data","networks",bertBaseLocation,modelName};
end

function supportfileName = userInputToSupportFileName(name)
persistent map;
if isempty(map)
    names = namesArray();
    map = containers.Map(names(:,1),names(:,2));
end
supportfileName = map(name);
end

function names = namesArray()
names = [
    "base",               "uncased_L12_H768_A12";
    "multilingual-cased", "multicased_L12_H768_A12";
    "medium",             "uncased_L8_H512_A8";
    "small",              "uncased_L4_H512_A8";
    "mini",               "uncased_L4_H256_A4";
    "tiny",               "uncased_L2_H128_A2";
    "japanese-base-wwm",  "";
    "japanese-base",      ""];
end