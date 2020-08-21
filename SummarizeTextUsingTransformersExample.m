%% Summarize Text Using Transformers
% This example shows how to summarize a piece of text using GPT-2.
% 
% Transformer networks such as GPT-2 can be used to summarize a piece of
% text. The trained GPT-2 transformer can generate text given an initial
% sequence of words as input. The model was trained on comments left on
% various web pages and internet forums.
% 
% Because lots of these comments themselves contain a summary indicated by
% the statement "TL;DR" (Too long, didn't read), you can use the
% transformer model to generate a summary by appending "TL;DR" to the input
% text. The |generateSummary| function takes the input text, automatically
% appends the string |"TL;DR"| and generates the summary.

%% Load Transformer Model
% Load the GPT-2 transformer model using the |gpt2| function.

mdl = gpt2;

%% Load Data
% Extract the help text for the |eigs| function.

inputText = help('eigs')

%% Generate Summary
% Summarize the text using the |generateSummary| function.

rng('default')
summary = generateSummary(mdl,inputText)
