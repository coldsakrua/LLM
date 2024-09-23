# Lab 1: Decoder from Scratch

本lab需要实现一个decoder-only transformer模型，并完成训练和文本生成任务。代码主要部分已经给出，需要补充关键函数/代码的实现。

Lab主要分成三个部分，分别是模型的实现(`src/model.py`)、模型的训练(`src/train.py`)、文本生成(`src/generate.py`)。Lab的评测包括两部分，一是lab中包含的测试点，你可以在项目目录下运行 `pytest` 命令来测试你的lab代码，也可以运行 `pytest tests/test_x.py` 单独测试 `src/x.py`；二是模型的训练和文本生成推理实验，这个没有标准答案，可以自由选择模型的规模，只要提供的训练、推理代码可以正确运行得到需要的结果即可。


## Step 0: Python环境搭建

1. 安装conda。Conda是常用的python虚拟环境管理工具，可以建立多个互相隔离的python环境，安装的包互不干扰。
```bash
bash setup-conda.sh && source ~/.bashrc
```
2. 创建conda虚拟环境。 
```bash
conda create -n hw1 python=3.10
conda activate hw1
pip install -r requirements.txt
pip install -e .
```
3. 运行 `wandb login` 以设置在实验中追踪weights & biases（你需要一个[weights & biases账户](https://wandb.ai/login)）。
4. 下载预先经过tokenizer处理的训练数据集。
```bash
curl https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz -o data/tokens.npz -L
```
5. 执行 `pytest tests/test_env.py` 确认你已经成功配置python环境。


### Step 0.5(optional): 关于einops 

你可以在lab中使用einops包。使用einops只是可选项，能用到的地方同样可以用pytorch完成，不过它可以让一些tensor操作更简单和直观。以下给出关于 `einops.rearrange` 的两个例子，你也许可以在lab中使用这个函数。关于einops的更多教程可以参考[这个](https://einops.rocks/1-einops-basics/)。

```
B, S, V = 16, 256, 50257

x = torch.rand(B, S, V)

# flatten axes B and V
y = einops.rearrange(x, "B S V -> (B S) V")
y = x.reshape(B * S, V)

# transpose axes S and V
z = einops.rearrange(x, "B S V -> B V S")
z = torch.permute(x, (0, 2, 1))
```


## Step 1: Decoder-only transformer

你需要实现一个decoder-only transformer模型。在 `src/model.py` 中提供了一个整体框架，里面包含所有需要的类和函数的声明。**请不要修改类和函数名以及他们的参数。请不要import新的python包。这会影响自动化评测过程。**注意一下代码的效率问题，可以调用pytorch函数时就调用，而不是自己写实现。请不要调用官方实现好的类和函数来代替你的具体实现过程，例如torch.nn.TransformerDecoder。如果你不确定一个类或函数的调用是否允许，可以问一下助教。

代码中有四个类需要你实现：
1. MultiHeadAttention - 实现Masked Multi-Head Attention模块
2. FeedForward - 实现Feed Forward模块
3. DecoderBlock - 一层decoder。由于我们实现的是decoder-only transformer，因此不需要实现Encoder-Decoder Multi-Head Attention。
4. DecoderLM - 完整的模型，包括embedding步骤，多个decoder blocks，以及最终的output logits。


## Step 2: 训练transformer

你需要基于Step 1实现的模型进行训练。训练所用的数据集在Step 0的第四步已经下载完成，这是[C4 corpus](https://huggingface.co/datasets/allenai/c4)的一个子集，而C4数据集又是[Common Crawl web corpus](https://commoncrawl.org/)的一个子集。

`src/train.py`包含代码整体框架。你需要实现下面5个函数：
1. train - 主要的训练循环
2. random_batch_sampler - 一个训练用的数据采样函数，产生随机打乱的一批数据
3. sequential_batch_sampler - 一个校验用的数据采样函数，产生连续的数据
4. cosine_lr_schedule - 使用余弦退火的学习率调度器
5. compute_language_modeling_loss - 训练和评估时使用的损失函数

### 余弦退火
带热身的余弦退火是一种模型训练中使用的动态学习率策略。它包含两个阶段，在热身阶段，$t \in [0, num\_warmup\_steps)$，学习率从0线性增长到$lr_{max}$（在$t=num\_warmup\_step$时达到$lr_{max}$）；在退火阶段，$t \in [num\_warmup\_steps, num\_training\_steps)$，学习率从$lr_{max}$衰减到$lr_{min}$，衰减过程遵循半余弦曲线（在$t=num\_training\_step$时达到$lr_{min}$）。当$t \geq num\_training\_steps$，学习率保持在$lr_{min}$。


### 配置文件

我们使用YAML格式文件来配置模型训练。你可以参考如下例子来探索超参数的调整。

```yaml
output_dir: outputs/GPT-tiny  # <- where the output files are written
tokenizer_encoding: gpt2      # <- the tokenizer encoding, used by tiktoken (YOU SHOULD NOT CHANGE THIS)
model_config:
  n_embd: 32                  # <- dimension of token and positional embeddings 
  n_head: 2                   # <- number of attention heads in multihead attention
  n_positions: 128            # <- the maximum number of tokens that the model can take
  n_layer: 2                  # <- number of decoder blocks
device: auto                  # <- which device to put the model on (YOU DO NOT NEED TO CHANGE THIS)
batch_size: 32                # <- number of sequences to feed into the model at a time
seq_len: 128                  # <- length of each sequence in training and evaluation, <= model_config.n_positions
num_warmup_steps: 10          # <- number of warmup steps in cosine annealing
num_training_steps: 2000      # <- number of training steps in cosine annealing
grad_accumulation_steps: 1    # <- number of micro steps of gradient accumulation before every model update
min_lr: 1e-4                  # <- minimum learning rate in cosine annealing
max_lr: 5e-4                  # <- maximum learning rate in cosine annealing
```

### 训练模型

在实现 `src/model.py` 和 `src/train.py` 之后且通过pytest校验之后，你还需要自己训练一个模型。作为样例，你可以运行 `python src/train.py configs/GPT-tiny.yaml` 来训练一个小模型（～2M参数）。

你可以创建新的配置文件来配置其他超参数，训练你的最终模型。最终提交的模型validation perplexity (PPL)不超过50即可得到满分。鼓励大家探索PPL更低的配置，但不要超过分配的机时。

## Step 3. 文本生成
`src/generate.py` 提供了文本生成的主要代码。为了生成文本，需要为LLM提供prompt，在data/prefixs.json中提供了prompt输入。需要实现的函数有两个：

1. generate - 给定一个DecoderLM和一系列prompts，生成tokens
2. softmax_with_temperature - 将一系列logits使用带temperature的softmax函数转化为生成token的概率。

在实现的函数通过pytest测试之后，你需要使用在Step 2中训练的模型对data/prefixs.json中的输入生成文本。你可以探索不同的temperature值对生成结果的影响。这个步骤不需要达到任何分数要求，你只需要给一个你生成的结果即可:)

### Tokenizer

在文本生成任务中，输入和期望的输出都是文本字符串。在模型的推理过程中，输入和输出的token都用一个ID表示。Tokenizer实现单词字符串和token ID间的相互转换。在Lab中使用tiktoken库得到tokenizer，可调用encode/decode函数实现文本和ID的转换。具体用例可参考[这个](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)。

## 最终提交

你需要提交源代码(`src/model.py`, `src/train.py`, `src/generate.py`)，用于训练的超参数配置文件(config.yaml)，训练得到的模型参数(model.pt)，校验结果文件(eval.json)，文本生成结果(generation.jsonl)。

评分标准为：pytest中的每个case 1分；训练模型的校验结果的给分公式$6 * min(1, max(0, (1000-PPL)/950))$；文本生成结果给出即可得到3分。满分一共20分。

## Acknowledgement

Lab代码基本来自CMU LLM课程的[HW2](https://2023.cmu-llms.org/homework2/)。请同学不要把完成的Lab代码上传到开源代码平台。

下面是原Lab的Acknowledgement。

This code contains modifications from [nanoGPT](https://github.com/karpathy/nanoGPT)
([license](copyright/nanoGPT)) and [PyTorch](https://pytorch.org/)
([license](copyright/pytorch)).







