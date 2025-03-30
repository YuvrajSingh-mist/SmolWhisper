import argparse
from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    pre_trained_model_dir: str = None
    fine_tuned_model_dir: str = None
    epochs: int = 2
    block_size: int = 64
    batch_size: int = 256
    tgt_vocab_size: int = len(tokenizer)   
    embeddings_dims: int = 512
    attn_dropout: float = 0.1
    no_of_heads: int = 4 
    dropout: float = 0.1
    max_lr: float = 1.5e-3
    no_of_decoder_layers: int = 6 
    weight_decay_optim: float = 0.1
    log_mel_features: int = 80
    kernel_size: int = 3
    stride: tuple = (2,10)
    sr: int = 16000
    device: str = 'cuda:0'
    SAMPLING_RATE: int = 16000
    N_MELS: int = 80  
    WINDOW_DURATION: float = 0.025  # 25 milliseconds
    STRIDE_DURATION: float = 0.010  # 10 milliseconds
    max_t: int = 500
    n_channels: int = N_MELS
    clip: float = 1.0
    use_flash_attention: bool = True
    use_liger: bool = True
    use_torch_compile: bool = False 
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    eps: float = 1e-6
    beta_1: float = 0.9
    beta_2: float = 0.98
    save_checkpoint_dir: str = "checkpoints"

    save_checkpoint_iter: int = 50
    total_iters: int = 20000
    eval_iters: int = 50
    eval_check: int = 50
    warmup_iters: int = 700
    min_lr: float = 3e-6
    lr_decay_iters: int = 20000
    total_batch_size: int = 32768
    micro_batch_size: int = ModelArgs.batch_size
    gradient_accumulation_steps: int = total_batch_size // (micro_batch_size * (ModelArgs.block_size * torch.cuda.device_count()))