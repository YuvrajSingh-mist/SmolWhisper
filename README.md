
# Introducing SmolWhisper - A Small ASR Model  

- So, I trained a Whisper model a ~30M (whisper tiny.en) architecture I coded from ground up to build a small ASR model, going through the below-mentioned stage from scratch.
- Trained on GigaSpeech dataset form HuggingFace consisting of 250 hours of clean audio for a total of full 3 epochs


### 1) Pretraining


#### Dataset

 - I used the [GigaSpeech]([https://huggingface.co/datasets/HuggingFaceFW/fineweb?row=0](https://huggingface.co/datasets/speechcolab/gigaspeech)) dataset from HuggingFace ('s' checkpoint) consisting of roughly 250 hrs.

  1) Train dataset - 200k rows
  2) Val dataset - 4k rows


---

####  ModelArgs (Hyperparameters)

## Model Architecture
| Parameter               | Description                     | Default Value | Range          |
|-------------------------|---------------------------------|---------------|----------------|
| `block_size`            | Context window length          | 256           | 64-4096        |
| `embeddings_dims`       | Hidden dimension size          | 512           | 128-4096       |
| `no_of_heads`          | Attention heads                | 8             | 4-32           |
| `no_of_decoder_layers` | Transformer layers             | 16            | 4-64           |
| `experts`              | Total MoE experts              | 8             | 2-128          |
| `top_experts`          | Active experts per token       | 2             | 1-4            |

## Training Configuration
| Parameter               | Description                     | Default Value | Typical Range   |
|-------------------------|---------------------------------|---------------|-----------------|
| `batch_size`           | Per-GPU batch size             | 128           | 32-1024         |
| `total_batch_size`     | Global batch size              | 524288        | 32k-2M          |
| `max_lr`              | Peak learning rate             | 6e-4          | 1e-5 to 1e-3    |
| `min_lr`              | Minimum learning rate          | 6e-5          | 1e-6 to 1e-4    |
| `weight_decay_optim`  | AdamW weight decay             | 0.1           | 0.0-0.2         |
| `clip`                | Gradient clipping              | 1.0           | 0.1-5.0         |

## Optimization Schedule
| Parameter               | Description                     | Default Value | Notes           |
|-------------------------|---------------------------------|---------------|-----------------|
| `warmup_iters`         | LR warmup steps                | 700           | ~5% of total iters |
| `lr_decay_iters`       | LR decay duration              | 20000         | =total_iters    |
| `gradient_accumulation_steps` | Micro-batch steps      | 4096          | total_batch_size/(batch_size*n_gpus) |

## Regularization
| Parameter               | Description                     | Default Value | Effect          |
|-------------------------|---------------------------------|---------------|-----------------|
| `dropout`              | General dropout rate           | 0.1           | 0.0-0.3         |
| `attn_dropout`         | Attention dropout              | 0.1           | 0.0-0.3         |


---
### Hardware Setup

 - Used DPP using Pytorch torchrun consisting of 2x A100s SXM (80GB VRAM each) rented on runpod.io

---

#### Frameworks:
**Pytorch**

--- 

#### Epochs/Steps
- Epochs (train) = 3 (gradient accumulation of 1)

- Val iterations = every 50 steps
---

#### Losses

 - Result - Pretraining  

   Train loss: 3.16  
   Val Loss: 3.93  


---

#### Screenshots of the loss curves

- Pretrain

![Train Loss Curves](images/loss.jpg)


--- 


### Local setup


### Requirements



```python
git [clone the repo](https://github.com/YuvrajSingh-mist/SmolWhisper.git)
cd SmolWhisper
bash ./install.sh

```
- A wandb.ai account for plotting graphs for your loss curves

- On your terminal run
```python
wandb login
```

- Enter the api key and follow the instructions and once you are succesfully logged in follow the given steps


---

### Running 


#### Training a model

- Kindly hange 'device' to any of your available cuda gpus.

To run:

```python
cd SmolWhisper
```

Prepare the dataset (gigaspeech)

```python

python data.py 


```

```python
bash ./install.sh
```


Train the model

Now run the following command 

```python
torchrun --standalone --nproc_per_node=gpu trainer.py \  
   --epochs 4 \
--block_size 256 \
--batch_size 128 \
--embeddings_dims 512 \
--attn_dropout 0.1 \
--no_of_heads 8 \
--dropout 0.1 \
--val_epochs 2 \
--max_lr 6e-4 \
--no_of_decoder_layers 16 \
--weight_decay_optim 0.1 \
--beta_1 0.9 \
--beta_2 0.95 \
--clip 1.0 \
--device cuda \
--vocab_size 50304 \
--eps 1e-5 \
--dtype "bfloat16" \
--save_checkpoint_dir "checkpoints" \
--prompt "Once upon a time" \
--save_checkpoint_iter 50 \
--total_iters 20000 \
--eval_iters 50 \
--eval_check 100 \
--warmup_iters 700 \
--min_lr 6e-5 \
--lr_decay_iters 20000 \
--total_batch_size 524288 \
--micro_batch_size 128 \
--gradient_accumulation_steps 4096 \
--experts 8 \
--top_experts 2 \
--use_flash_attention True \
--use_liger True \
--use_compile False \
--use_checkpointing False \
--noisy_topk True
```


#### Inference on a pretrained model

 
```python
python inference.py --audio_path "YOUR_PATH_HERE" --max_length 128 --temperature 0.8  --model_path "TRAINED_MODEL_PATH_HERE"
```


