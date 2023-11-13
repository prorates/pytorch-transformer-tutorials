# Pytorch Transformer Tutorials

This repository contains different versions of Transformers implemented during tutorial using Pytorch.
The github, video, documentation are listed bellow.
This repository attempts to get those tutorials to work with new datasets.

## Model 1

This is based on the following [video](https://youtu.be/ISNdQcPhsts)
The code is original code is available [here](https://github.com/hkproj/pytorch-transformer)

## Model 2

The code was copy pasted from [here](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)
The same code seems available [here](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)

## Model 3

The source code seems to be [here](https://github.com/SamLynnEvans/Transformer?ref=blog.floydhub.com)

## Model 4

Builld a GPT from scratch [video][https://youtu.be/kCc8FmEb1nY]

## Model 5

The source code is based on the official pytorch Transformer. Code modified just for tutorial purpose.

## Model 6

This is based on the following [video](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)
The code is original code is available [here](https://github.com/ajhalthor/Transformer-Neural-Network)

## Model 7

See [Huggineface Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## Model 8

See [video](https://youtu.be/kCc8FmEb1nY)
The colab repo is [here](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

## Size requirements

This is coming from [here](https://schartz.github.io/blog/estimating-memory-requirements-of-transformers/)

### xxx
total_memory = memory_modal + memory_activations + memory_gradients


### Estimating model's memory requirements
Lets take GPT as an example. GPT consists of a number of transformer blocks (let's call it n_tr_blocks from now on). Each transformer block consists of following structure:

```
multi_headed_attention --> layer_normalization --> MLP -->layer_normalization
```

Each multi_headed_attention element consists of value nets, key and query. Let's say that each of these have n_head attention heads and dim dimensions. MLP also has a dimension of n_head * dim. The memory needed to store these will be

```
total_memory = memory of multi_headed_attention + memory of MLP
			 = memory of value nets + memory of key + memory of query + memory of MLP
			 = square_of(n_head * dim) + square_of(n_head * dim) + square_of(n_head * dim) + square_of(n_head * dim)
			 = 4*square_of(n_head * dim)
```
Since our modal contains n_tr_blocks units of these blocks. Total memory required by the modal becomes.

```
memory_modal = 4*n_tr_blocks*square_of(n_head * dim)
```

Above estimation does not take into account the memory required for biases, since that is mostly static and does not depend on things like batch size, input sequence etc.


### Estimating model activation's memory requirements

Multi headed attention is generally a softmax. More specifically it can written as:

```
multi_headed_attention = softmax(query * key * sequence_length) * value_net
```
query key and value_net all have a tensor shape of

```
[batch_size, n_head, sequence_length, dim]
```

query * key * sequence_length operation gives following resultant shape:

```
[batch_size, n_head, sequence_length, sequence_length]
```
This finally gives the memory cost of activation function as

```
memory_softmax  = batch_size * n_head * square_of(sequence_length)
```


query * key * sequence_length operation multiplied by value_net has the shape of [batch_size, n_head, sequence_length, dim]. MLP also has the same shape. So memory cost of these operations become:

```
memory of MLP  = batch_size * n_head * sequence_length * dim
memory of value_net  = batch_size * n_head * sequence_length * dim
```

This gives us the memory of model activation per block:

mem_act = memory_softmax + memory_value_net + memory_MLP
		= batch_size * n_head * square_of(sequence_length)
		  + batch_size * n_head * sequence_length * dim
		  + batch_size * n_head * sequence_length * dim
		= batch_size * n_head * sequence_length * (sequence_length + 2*dim)
Memory of model activation across the model will be:

```
n_tr_blocks * (batch_size * n_head * sequence_length * (sequence_length + 2*dim))
````

### Summing it all up
To sum up total memory needed for fine-tuning/training transformer models is:

```
total_memory = memory_modal + 2 * memory_activations
```
Memory for modal is:

```
memory_modal = 4*n_tr_blocks*square_of(n_head * dim)
```
And memory for model activations is:

```
n_tr_blocks * (batch_size * n_head * sequence_length * (sequence_length + 2*dim))
```
These rough formulas can be written more succintly using following notation.

```
R = n_tr_blocks = number of transformer blocks in the model
N = n_head = number of attention heads
D = dim = dimension of each attention head
B = batch_size = batch size
S = sequence_length = input sequence length

memory modal = 4 * R * N^2 * D^2

memory activations = RBNS(S + 2D)
```
Total memory consumption if modal training is

```
M = (4 * R * N^2 * D^2) + RBNS(S + 2D)
```
If we have a very long sequence lengths S >> D S + 2D <--> S hence M in this case becomes:

```
M = (4 * R * N^2 * D^2) + RBNS(S) = 4*R*N^2*D^2 + RBNS^2

M is directly proportional to square of length of input sequence for large sequences
M is lineraly proportional to the batch size.
```

#### XXX

These rough formula for estimating the memory requirements of fine tuning transformer models

```
R = n_tr_blocks = number of transformer blocks in the model
N = n_head = number of attention heads
D = dim = dimension of each attention head
B = batch_size = batch size
S = sequence_length = input sequence length


memory modal = 4 * R * N^2 * D^2

memory activations = RBNS(S + 2D)

total memory required = ((4 * R * N^2 * D^2) + RBNS(S + 2D)) * float64 memory in bytes
```

Insights
Memory consumption is directly proportional to square of length of input sequence for large sequences

Memory consumption is lineraly proportional to the batch size.

