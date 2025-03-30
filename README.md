
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


| Parameter                      | Description                                                                 | Default Value                     | Type      |
|--------------------------------|-----------------------------------------------------------------------------|-----------------------------------|-----------|
| `epochs`                       | Number of training epochs                                                   | `4`                               | `int`     |
| `block_size`                   | Size of each block (context length)                                         | `512`                             | `int`     |
| `batch_size`                   | Batch size for training                                                    | `64`                              | `int`     |
| `inference`                    | Inference mode (not specified)                                              | `None`                            | `None`    |
| `embeddings_dims`              | Dimensionality of embeddings                                                | `512`                             | `int`     |
| `attn_dropout`                 | Dropout rate for attention layers                                           | `0.1`                             | `float`   |
| `no_of_heads`                  | Number of attention heads                                                   | `8`                               | `int`     |
| `dropout`                      | Dropout rate for the model                                                  | `0.1`                             | `float`   |
| `val_epochs`                   | Number of validation epochs                                                 | `2`                               | `int`     |
| `max_lr`                       | Maximum learning rate                                                       | `6e-4`                            | `float`   |
| `no_of_decoder_layers`         | Number of decoder layers                                                    | `8`                               | `int`     |
| `weight_decay_optim`           | Weight decay for the optimizer                                              | `0.1`                             | `float`   |
| `beta_1`                       | Beta 1 for Adam optimizer                                                   | `0.9`                             | `float`   |
| `beta_2`                       | Beta 2 for Adam optimizer                                                   | `0.95`                            | `float`   |
| `clip`                         | Gradient clipping value                                                     | `1.0`                             | `float`   |
| `device`                       | Device to run the model (`cuda` or `cpu`)                                   | `'cuda'`                          | `str`     |
| `no_kv_heads`                  | Number of key-value heads                                                   | `2`                               | `int`     |
| `vocab_size`                   | Size of the vocabulary                                                      | `50304`                           | `int`     |
| `eps`                          | Epsilon value for numerical stability                                       | `1e-5`                            | `float`   |
| `dtype`                        | Data type for tensors (`bfloat16` if supported, else `float16`)             | `'bfloat16'` or `'float16'`       | `str`     |
| `save_checkpoint_dir`          | Directory to save model checkpoints                                         | `"checkpoints"`                   | `str`     |
| `prompt`                       | Default prompt for inference                                                | `"Once upon a time"`              | `str`     |
| `save_checkpoint_iter`         | Save checkpoint every N iterations                                         | `50`                              | `int`     |
| `total_iters`                  | Total number of training iterations                                        | `10000`                           | `int`     |
| `eval_iters`                   | Evaluate model every N iterations                                          | `50`                              | `int`     |
| `eval_check`                   | Check evaluation metrics every N iterations                                | `100`                             | `int`     |
| `warmup_iters`                 | Number of warmup iterations for learning rate scheduling                   | `700`                             | `int`     |
| `min_lr`                       | Minimum learning rate (10% of `max_lr`)                                     | `0.1 * max_lr`                    | `float`   |
| `lr_decay_iters`               | Number of iterations for learning rate decay                               | `10000`                           | `int`     |
| `total_batch_size`             | Total batch size across all devices                                         | `524288`                          | `int`     |
| `micro_batch_size`             | Micro batch size per device                                                | `batch_size`                      | `int`     |
| `gradient_accumulation_steps`  | Gradient accumulation steps                                                 | `total_batch_size // (micro_batch_size * (block_size * torch.cuda.device_count()))` | `int` |
| `no_kv_heads`                  | Number of key-value heads                                                   | `2`                               | `int`     |
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


