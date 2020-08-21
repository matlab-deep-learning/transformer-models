# Transformer Models for MATLAB
This repository implements deep learning transformer models in MATLAB.

## Requirements
* MATLAB R2020a or later
* Deep Learning Toolbox

## Getting Started
Download or [clone](https://www.mathworks.com/help/matlab/matlab_prog/use-source-control-with-projects.html#mw_4cc18625-9e78-4586-9cc4-66e191ae1c2c) this repository to your machine and open it in MATLAB.

## Functions
### gpt2
`mdl = gpt2` loads a pretrained GPT-2 transformer model and if necessary, downloads the model weights.

### generateSummary
`summary = generateSummary(mdl,text)` generates a summary of the string or `char` array `text` using the transformer model `mdl`. The output summary is a char array.

`summary = generateSummary(mdl,text,Name,Value)` specifies additional options using one or more name-value pairs.

* `'MaxSummaryLength'` - The maximum number of tokens in the generated summary. The default is 50.
* `'TopK'` - The number of tokens to sample from when generating the summary. The default is 2.
* `'Temperature'` - Temperature applied to the GPT-2 output probability distribution. The default is 1.
* `'StopCharacter'` - Character to indicate that the summary is complete. The default is `'.'`.

## Example: Summarize Text Using GPT-2
The example `SummarizeTextUsingTransformersExample.m` shows how to summarize a piece of text using GPT-2.

Transformer networks such as GPT-2 can be used to summarize a piece of text. The trained GPT-2 transformer can generate text given an initial sequence of words as input. The model was trained on comments left on various web pages and internet forums.

Because lots of these comments themselves contain a summary indicated by the statement "TL;DR" (Too long, didn't read), you can use the transformer model to generate a summary by appending "TL;DR" to the input text. The `generateSummary` function takes the input text, automatically appends the string `"TL;DR"` and generates the summary.

### Load Transformer Model
Load the GPT-2 transformer model using the `gpt2` function.

```matlab:Code
mdl = gpt2;
```

### Load Data
Extract the help text for the `eigs` function.

```matlab:Code
inputText = help('eigs');
```

### Generate Summary
Summarize the text using the `generateSummary` function.

```matlab:Code
rng('default')
summary = generateSummary(mdl,inputText)
```

```text:Output
summary =

    '    EIGS(AFUN,N,FLAG) returns a vector of AFUN's n smallest magnitude eigenvalues'
```