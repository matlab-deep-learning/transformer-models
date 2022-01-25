# MATLABのためのトランスフォーマーモデル
[![CircleCI](https://img.shields.io/circleci/build/github/matlab-deep-learning/transformer-models?label=tests)](https://app.circleci.com/pipelines/github/matlab-deep-learning/transformer-models)

*README in English is [here](https://github.com/matlab-deep-learning/transformer-models/blob/master/README.md)

このリポジトリは、MATLABで深層学習トランスフォーマーモデルを実装するための関数や例題が含まれています。なお、BERT、finBERT、GPT-2は、基本的には、BERTの"multilingual-cased"を除き英語の文書を用いて学習されており、入力値も英語であることが想定されています。


## 要件
### BERT・FinBERT
- MATLABバージョン R2021a以降
- Deep Learning Toolbox
- Text Analytics Toolbox

### GPT-2
- MATLABバージョン R2020a以降
- Deep Learning Toolbox

## はじめに
このリポジトリをダウンロードするか、お手元のマシンに[クローン](https://www.mathworks.com/help/matlab/matlab_prog/use-source-control-with-projects.html#mw_4cc18625-9e78-4586-9cc4-66e191ae1c2c)してMATLABで開きます。

## 関数の説明
### bert
`mdl = bert` により、学習済みBERTトランスフォーマーモデルが読み込まれます。初回実行時などは、必要に応じてモデルの重みもダウンロードされますので時間がかかります。出力値 `mdl` は、BERT用トークナイザ― `Tokenizer` と、モデルパラメータ `Parameters` を含む構造体です。

`mdl = bert("Model",modelName)` により、使用するBERTモデルの種類を指定します。：

- `"base"` (既定値) - 12-layer、768-hidden
- `"multilingual-cased"` - 12-layer、768-hidden。トークナイザーでは大文字と小文字が区別されます。このモデルは、多言語データで学習されています。
- `"medium"` - 8-layer、512-hidden 
- `"small"` - 4-layer、512-hidden
- `"mini"` - 4-layer、256-hidden
- `"tiny"` - 2-layer、128-hidden

### bert.model
`Z = bert.model(X,parameters)` は、指定されたパラメータを持つ符号化トークンの入力 `1`-by-`numInputTokens`-by-`numObservations` 配列上で BERT モデルによる推論を実行します。 出力 `Z` は、サイズ (`NumHeads*HeadSize`)-by-`numInputTokens`-by-`numObservations` の 配列です。要素 `Z(:,i,j)` は、入力トークン `X(1,i,j)` のBERT埋め込みに相当します。

`Z = bert.model(X,parameters,Name,Value)` は、1 つ以上の名前と値のペアの引数で指定された追加オプションを設定します。：

- `"PaddingCode"` パディングトークンに対応する正の整数。デフォルトは `1` です。
- `"InputMask"` - `X`  と同じサイズの論理配列，あるいは空の配列として指定します．このマスクは，パディングに対応するインデックス位置では偽でなければならず，それ以外の場所では真でなければならない．マスクが `[]` である場合，関数は PaddingCode の名前と値の組にしたがってパディングを決定します．デフォルトは `[]` です． 
- `"DropoutProb"` - 出力活性化に対するドロップアウトの確率。デフォルトは `0` です。
- `"AttentionDropoutProb"` - アテンション層で利用されるドロップアウトの確率。デフォルトは `0` です。
- `"Outputs"` - 出力を返すレイヤーのインデックス、正の整数のベクトル、または `"last"` として指定されます。`"Outputs"` が `"last"` の場合、この関数は最後のエンコーダ層からの出力のみを返します。デフォルトはが `"last"` です。
- `"SeparatorCode"` - 正の整数で指定されるセパレータトークン。デフォルトは `103` です。

### finbert
`mdl = finbert` は、金融テキストのセンチメント（感情）分析のために、事前に学習されたBERTトランスフォーマーモデルを読み込みます。出力される `mdl` は、フィールド `Tokenizer` と `Parameters` を持つ構造体で、それぞれ BERTトークナイザーとモデルパラメータを含んでいます。

`mdl = finbert("Model",modelName)` にて、使用するFinBERTモデルの種類を指定します。：
- `"sentiment-model"` (既定値) - ファインチューニングされたセンチメント分類モデル。
- `"language-model"` - BERT-Base アーキテクチャを使用する、FinBERT で事前学習された言語モデル。

### finbert.sentimentModel
`sentiment = finbert.sentimentModel(X,parameters)` は、入力 `1`-by-`numInputTokens`-by-`numObservations` の各トークンを、指定されたパラメータで分類します。出力されるセンチメントは、カテゴリ `"positive"`, `"neutral"`, または `"negative"` からなるカテゴリ配列である。
`[sentiment, scores] = finbert.sentimentModel(X,parameters)` は、対応するセンチメントスコアを `[-1 1]` の範囲で返すこともできます。

### gpt2
`mdl = gpt2` は学習済み GPT-2 トランスフォーマーモデルを読み込み、初回実行時など、必要であればモデルの重みをダウンロードします。

### generateSummary
`summary = generateSummary(mdl,text)` は、トランスフォーマーモデル `mdl` を用いて、文字列または `char` 配列 `text` の要約を生成します。出力される要約は char 配列である。

`summary = generateSummary(mdl,text,Name,Value)` は、1 つ以上の名前と値のペアの引数で指定された追加オプションを設定します。：

* `"MaxSummaryLength"` - 生成されるサマリーのトークン数の最大値です。デフォルトは50です。
* `"TopK"` - 要約を生成する際にサンプリングするトークンの数です。デフォルトは 2 です。
* `"Temperature"` - GPT-2出力確率分布に適用される温度。デフォルトは1です。
* `"StopCharacter"` - 要約が完了したことを示す文字。デフォルトは `"."` です。

## 例題：BERTによるテキスト分類
事前学習済みBERTモデルの最も単純な使用方法は、特徴抽出器として使用することです。BERT モデルにより文書を特徴ベクトルに変換し、それを入力として使用して、深層学習分類ネットワークを学習することができます。

例題 [`ClassifyTextDataUsingBERT.m`](./ClassifyTextDataUsingBERT.m) では、工場レポートのデータセットを与えられた故障原因を分類するために事前学習済み BERT モデルを使用する方法を示しています。

## 例題：学習済みBERTモデルのファイン チューニング
事前学習済みBERT モデルを最大限に活用するために、タスクに応じた BERT パラメータの重みを再学習し、微調整することが可能です。（＝ファインチューニング）

例題 [`FineTuneBERT.m`](./FineTuneBERT.m) では、工場レポートのデータセットが与えられたときに、故障原因を分類するために事前学習済みの BERT モデルを微調整する方法を示しています。

## 例題：FinBERTによる感情分析
FinBERTは、金融テキストデータで学習し、センチメント分析用にファインチューニングされたセンチメント分析モデルです。

例題 [`SentimentAnalysisWithFinBERT.m`](./SentimentAnalysisWithFinBERT.m) では、事前学習済み FinBERT モデルを使って、金融ニュースレポートの感情分類を行う方法を示しています。

## 例題：BERTおよびfinBERTによるマスク語予測
BERTモデルは、様々なタスクを実行するために学習されている。BERTによって成し遂げられるタスクの一つは、マスク値（[MASK]値）で置換された、テキスト中のトークンを予測するタスクです。Masked Language Modelとしても知られています。

例題 [`PredictMaskedTokensUsingBERT.m`](./PredictMaskedTokensUsingBERT.m) では、事前学習済み BERT モデルを使用して、マスクされたトークンを予測し、トークン確率を計算する方法を示しています。

例題 [`PredictMaskedTokensUsingFinBERT.m`](./PredictMaskedTokensUsingFinBERT.m) では、事前学習済み FinBERT モデルを使用して、金融テキストのマスクトークンを予測する方法と、トークン確率を計算する方法を示しています。

## 例題：GPT-2による文書要約
GPT-2などのトランスフォーマーネットワークは、テキストの一部を要約するために使用することができます。学習済みGPT-2トランスフォーマーは、最初の単語列を入力としてテキストを生成することが可能です。このモデルは、様々なウェブページやインターネットフォーラムに残されたコメントに対して学習させたものです。

これらのコメントの多くには「TL;DR」（Too long, didn't read）という文で示される要約が含まれているので、変換器モデルを使って、入力テキストに「TL;DR」を付加して要約を生成することができます。関数 `generateSummary` は、入力テキストを受け取って、文字列 `"TL;DR"` を自動的に付加し、要約を生成します。

例題 [`SummarizeTextUsingTransformersExample.m`](./SummarizeTextUsingTransformersExample.m) では、GPT-2を用いてテキストを要約する方法を紹介しています。
