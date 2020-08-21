function summary = generateSummary(mdl, text, nameValueArguments)
% GENERATESUMMARY   Generate summary of text with GPT-2
%
%   summary = GENERATESUMMARY(mdl, text) generates a summary of the
%   string or char array text using the transformer model mdl. The output
%   summary is a char array.
%
%   summary = GENERATESUMMARY(mdl, text, 'PARAM1', 'VAL1', ...)
%   specifies optional name/value pairs for creating the summary:
%
%   'MaxSummaryLength'      - The maximum number of tokens in the generated
%                             summary. The default is 50.
%   'TopK'                  - The number of tokens to sample from when
%                             generating the summary. The default is 2.
%   'Temperature'           - Temperature applied to the GPT-2 output
%                             probability distribution. The default is 1.
%   'StopCharacter'         - If the model generates this character its
%                             summary is finished. The default is '.'.

%   Copyright 2020 The MathWorks, Inc.

arguments
    mdl
    text                                      {iIsScalarString}
    nameValueArguments.MaxSummaryLength (1,1) {mustBeInteger, mustBePositive} = 50
    nameValueArguments.TopK             (1,1) {mustBeInteger, mustBePositive} = 2
    nameValueArguments.Temperature      (1,1) {mustBePositive} = 1
    nameValueArguments.StopCharacter          {iIsScalarString} = '.'
end

% Unpack arguments
maxSummaryLength    = nameValueArguments.MaxSummaryLength;
topK                = nameValueArguments.TopK;
temperature         = nameValueArguments.Temperature;
stopCharacter       = nameValueArguments.StopCharacter;

% Remove newline tokens
inputText = replace(char(text), newline, char.empty());

% Get the GPT-2 tokenizer and model.
enc = mdl.Tokenizer;
parameters = mdl.Parameters;

% To instruct the GPT-2 network to generate a summary, we append TL;DR to
% the end of the text
inputText = [inputText ' TL;DR'];

% Encode some text
inputTokens = enc.encode(inputText);

% Ensure the text to be summarized fits within the context window of the
% model
if length(inputTokens) > (parameters.Hyperparameters.NumContext-maxSummaryLength-3)
    inputTokens = inputTokens(1:(parameters.Hyperparameters.NumContext-maxSummaryLength-3));
    inputText = enc.decode(inputTokens);
    inputText = [inputText ' TL;DR'];
    inputTokens = enc.encode(inputText);
end

% Initialize the cell array of pasts
pasts = cell(parameters.Hyperparameters.NumLayers,1);

% Feed in the input except for the last token
[~, presents] = gpt2.model( inputTokens(1:(end-1)), pasts, ...
    parameters );

% Initialize the previous token.
previousToken = inputTokens(end);

% Initialize the summary
summary = [];

% Generate the summary
for i = 1:maxSummaryLength
    % Now run the model for another step
    [logits, presents] = gpt2.model( ...
        previousToken, presents, parameters );
    
    % Apply the temperature to the last logit
    logits = logits./temperature;
    
    % Filter out all except the top K logits
    logits = sampling.topKLogits(logits, topK);
    
    % Apply softmax to get probabilities.
    probabilities = softmax(logits,'DataFormat','CTB');
    
    % Sample from a categorical distribution with the logits
    nextToken = sampling.sampleFromCategorical(extractdata(probabilities));
    
    % Set the previous token for the next iteration
    previousToken = nextToken;
    
    % Work out whether we need to start a new line
    textToPrint = enc.decode(nextToken);    
    
    % Grow the summary array
    summary = [summary textToPrint]; %#ok<AGROW>
    
    % Stop if we generate a stop character
    if textToPrint == stopCharacter
        break
    end
end
end

function iIsScalarString(s)
validateattributes(s,{'string','char'},{});
switch class(s)
    case "string"
        validateattributes(s,{'string'},{'scalar'});
    case "char"
        validateattributes(s,{'char'},{'row'});
end
end