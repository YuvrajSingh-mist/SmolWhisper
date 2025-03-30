
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch

import wandb

import torch.optim as optim


import os
from config import ModelArgs
from model import Whisper

from data import prepare_dataset
from tokenizer import Tokenizer
import jiwer

torch.set_float32_matmul_precision('high')

scaler = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))


save_checkpoint_iter = ModelArgs.save_checkpoint_iter
total_iters = ModelArgs.total_iters
eval_iters = ModelArgs.eval_iters
eval_check = ModelArgs.eval_check
warmup_iters = ModelArgs.warmup_iters
min_lr = ModelArgs.min_lr
lr_decay_iters = ModelArgs.lr_decay_iters
total_batch_size = ModelArgs.total_batch_size
micro_batch_size = ModelArgs.micro_batch_size
gradient_accumulation_steps = ModelArgs.gradient_accumulation_steps


class Trainer:
    
    def __init__(self, model_args):


        def setup(rank=None, world_size=None):
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = '12355'
            init_process_group("nccl")
            # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            
        self.model_args = model_args  
        self.tokenizer = Tokenizer().ready_tokenizer()
        setup()
        
    def cleanup(self):
        destroy_process_group()

    def _save_snapshot(self, model, optimizer, epoch, step, save_dir):
        snapshot = {}
        snapshot["MODEL_STATE"] = model.module.state_dict()
        snapshot["OPTIMIZER_STATE"]= optimizer.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["STEP_RUN"] = step
        torch.save(snapshot, os.path.join(save_dir, "snapshot.pt"))
        print(f"Epoch: {epoch} | step {step} | Training snapshot saved at snapshot.pt")

    
    def train(self):

        device = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(int(device))
    
        print(f"Start running DDP on rank {device}.")

        global batch_size
        
        if(device == 0):

        
        
    #         # Initialise run
            wandb.init(
                # entity = 'rajceo2031',
                            project = 'Whisper-DDP',
                            # config = CFG,
                            # save_code = True,
                            #group = 'ANN',
                            #job_type = 'train'
    )
        print("wand initialized")
        
        model = Whisper()
        
        # print(f"Model on device {device} is ready")
        print(f"Model on device {device} is ready")

        optimizer = optim.AdamW(model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim, eps=ModelArgs.eps, fused=True)
        
        if(ModelArgs.use_torch_compile):
            model = torch.compile(model)
        
        model = model.to(device)
        
        model = DDP(model, device_ids=[device])
        

    
        def compute_wer(reference, hypothesis):

            error = jiwer.wer(reference, hypothesis)
            return error 
        
        model.eval()
        world_size = torch.cuda.device_count()
        @torch.inference_mode()
        def estimate_loss(val_loader, val_iterator, device):
            out = {}
            # train_loader = prepare_dataset('train', batch_size)
            
            # val_loader_iterator = iter(val_loader)
            loader = None
            epoch_loss = None
            epoch_losses = []
            # print("Starting the eval...")
            for split in ['val']:
                print(f"Starting with {split} evaluation...")
                # losses = torch.zeros(val_epochs)
                # if(split == 'train'):
                #         loader = train_loader
                # if(split == 'val'):
                #         loader = val_loader
                for step in range(eval_check):  
                    try:
                        batch = next(val_iterator)
                    except StopIteration:
                        val_loader_iterator = iter(val_loader)
                        batch = next(val_loader_iterator)
                    
                    total_loss = 0  
                    # loader.sampler.set_epoch(step)
                    total_batches = 0 
                    # batch = next(val_loader_iterator)
                    # for batch in loader:  # Loop through DataLoader batches
                    idx = batch['input_ids'].to(device)
                    targets = batch['labels'].to(device)
                    spec = batch['spectrogram'].to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        
                        loss = model(spec, idx, actual_labels=targets)
                        # batch_size, block_size, embeddings_dims = logits.shape
                        # logits = logits.view(batch_size * block_size, embeddings_dims)  # Flatten tokens
                        # targets = targets.view(batch_size * block_size)

                        # loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                        total_loss += loss.item()
                        total_batches += 1

                # Compute mean loss for this epoch
                epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
                epoch_losses.append(epoch_loss)

                    # print(f"Epoch {epoch + 1}/{val_epochs}: Loss = {epoch_loss:.4f}")

                # Compute mean loss across all evaluation epochs
                out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                epoch_loss = None
                epoch_losses = []

            model.train()
            return out

        # model = model.to(rank)
        model.train()
        count = 0
    
        train_dataloader = prepare_dataset('train', device, batch_size)
        val_loader= prepare_dataset('val', device, batch_size)
        test_loader = prepare_dataset('test', device, None)

        print("Loaders ready both")
        # epochs = epochs

        train_loader_length = 0
        train_data_iterator = iter(train_dataloader)
        val_data_iterator = iter(val_loader)
        test_iter = iter(test_loader)
        token_count = 0
        if(device == 0):
            train_loader_length = len(train_dataloader)
        
        for epoch in range(ModelArgs.epochs):
            for step in tqdm(range(total_iters)):
            # print("Dataloader things: ", batch)
            # print("Total batches: ", len(train_dataloader))
                
                
                if(device == 0):
                    # if(step % 100 == 0):
                #     if(step == train_loader_length):
                #       break
                        print("Step : ", step, "/", total_iters)
                        print('Total batches: ', len(train_dataloader))
                        print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                        print("Total tokens processed: ", token_count)
                        
                # all_gpus_avg_train_loss = None
                # all_gpus_avg_val_loss = None
                # every once in a while evaluate the loss on train and val sets
                if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                    losses = estimate_loss( val_loader, val_data_iterator, 'cuda')
                    # avg_train_loss = losses['train']
                    avg_val_loss = losses['val']
                    # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    # if device == 0:  # Only print on main process
                    print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
   
                    avg_val_loss = torch.Tensor([losses['val']]).to(device)
                    # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    
                    if device == 0:
                        # all_gpus_avg_train_loss = avg_train_loss / world_size
                        # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                        all_gpus_avg_val_loss = avg_val_loss / world_size
                        print(f"All_GPUs_Val_losses: {all_gpus_avg_val_loss.item():.4f}")
                        
                    
                        wandb.log({
                            # "Learning Rate": optimizer.param_groups[0]['lr'],
                            # "All_GPUs_Train_losses": all_gpus_avg_train_loss,
                            "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                            # "training_step_loss": losses['train'],
                            "val_step_loss": losses['val'],
                            # "Step": step,
                            # "Epoch": epoch
                        })
                    
                    
                
            

                if step % save_chechpoint_iter == 0 and device == 0 and step != 0:
                    print(f"Saving the model checkpoint for step: {step}")
                    _save_snapshot(model, optimizer, None, None, step)
                
                accumulated_loss = 0.0
                
                
                optimizer.zero_grad(set_to_none=True)
                for micro_step in range(gradient_accumulation_steps):
                    try:
                        batch = next(train_data_iterator)
                    except StopIteration:
                        train_data_iterator = iter(train_dataloader)
                        batch = next(train_data_iterator)
            
                    idx = batch['input_ids'].to(device)
                    targets = batch['labels'].to(device)
                    spec = batch['spectrogram'].to(device)
                    token_count += len(idx)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = model(spec, idx, actual_labels=targets)
                      
                        loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
                        accumulated_loss += loss.detach()
                    
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
                    scaler.scale(loss).backward()
                        # Check for unused parameters
                    unused_params = find_unused_parameters(model)
                    if unused_params:
                        print(f"Unused parameters: {unused_params}")
                # break
            
                    if(device == 0):
                        if(micro_step % 10 == 0):
                    #     if(step == train_loader_length):
                    #       break
                            
                            print("Micro Batch : ", micro_step)
                            print("Step : ", step, "/", total_iters)
                            print('Total batches: ', len(train_dataloader))
                            print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                            print("Total tokens processed: ", token_count)
                    # count += 1
            
                lr = get_lr(step)
                for params in optimizer.param_groups:
                    params['lr'] = lr
                    
                
                
                # Compute gradient norms before clipping
                if(clip != 0.0):
                    scaler.unscale_(optimizer) #To avoid underflow
                    total_norm_before = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
                    )

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)

                    # Compute gradient norms after clipping
                    total_norm_after = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
                    )
                    
                    if(device  == 0 and step !=0):
                        print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                        print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

                scaler.step(optimizer)
                scaler.update()
            
                # optimizer.step()
                # new_scheduler.step()
                # reference = batch['text']
                # hypothesis = logits
                torch.cuda.synchronize() 
                torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                

                if(device == 0):
                    wandb.log({
                            "Learning Rate": lr,
                            "All_GPUs_Train_losses": accumulated_loss.item(),
                            # "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                            # "training_step_loss": losses['train'],
                            # "val_step_loss": losses['val'],
                            # "WER": wer,
                            "Step": step,
                            "Grad Norm": total_norm_before.item(),
                            # "Epoch": epoch
                            
                        })
                
                if device == 0 and step % 20 == 0:
                    count = 2
                    try:
                        batch = next(test_iter)
                    except StopIteration:
                        test_loader_iterator = iter(test_loader)
                        batch = next(test_loader_iterator)
                            
                    while(count):  
                        # prompt = "Once upon a time"
                        reference, generated_text = topk_sampling(model, batch, max_length=50, top_k=50, temperature=1.0, device=device)
                        wer = compute_wer(reference, generated_text)
                        # print(f" Step: {step} | Generated Text: {generated_text}")
                        print(f" Step: {step} | WER: {wer}")
                        wandb.log({
                            "Val WER": wer
                        })
            
                        print(f" Step: {step} | Generated Text: {generated_text} | Real Text: {reference}")

                
                        count -= 1
                
        
        if device == 0:
            wandb.finish()
        cleanup()
        
    


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Arguments")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=ModelArgs.epochs, help="Number of training epochs")
    # parser.add_argument("--block_size", type=int, default=ModelArgs.block_size, help="Context window size")
    parser.add_argument("--batch_size", type=int, default=ModelArgs.batch_size, help="Batch size per device")
    parser.add_argument("--micro_batch_size", type=int, default=ModelArgs.batch_size, help="Micro batch size per device")
    parser.add_argument("--total_batch_size", type=int, default=ModelArgs.total_batch_size, help="Total batch size across devices")
    
    # Model architecture
    parser.add_argument("--embeddings_dims", type=int, default=ModelArgs.embeddings_dims, help="Embedding dimension size")
    parser.add_argument("--no_of_heads", type=int, default=ModelArgs.no_of_heads, help="Number of attention heads")
    parser.add_argument("--no_of_decoder_layers", type=int, default=ModelArgs.no_of_decoder_layers, help="Number of decoder layers")
    
    # Optimization
    parser.add_argument("--max_lr", type=float, default=ModelArgs.max_lr, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=ModelArgs.min_lr, help="Minimum learning rate")
    parser.add_argument("--weight_decay_optim", type=float, default=ModelArgs.weight_decay_optim, help="Weight decay value")
    parser.add_argument("--clip", type=float, default=ModelArgs.clip, help="Gradient clipping threshold")
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=ModelArgs.dropout, help="Dropout probability")
    parser.add_argument("--attn_dropout", type=float, default=ModelArgs.attn_dropout, help="Attention dropout probability")
    
    # Training schedule
    parser.add_argument("--total_iters", type=int, default=ModelArgs.total_iters, help="Total training iterations")
    parser.add_argument("--warmup_iters", type=int, default=ModelArgs.warmup_iters, help="Warmup iterations")
    parser.add_argument("--lr_decay_iters", type=int, default=ModelArgs.lr_decay_iters, help="Learning rate decay iterations")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                      default=ModelArgs.gradient_accumulation_steps, 
                      help="Gradient accumulation steps")
    
    # Adam parameters
    parser.add_argument("--beta_1", type=float, default=ModelArgs.beta_1, help="Adam beta1 parameter")
    parser.add_argument("--beta_2", type=float, default=ModelArgs.beta_2, help="Adam beta2 parameter")
    parser.add_argument("--eps", type=float, default=ModelArgs.eps, help="Adam epsilon value")
    
    # System configuration
    parser.add_argument("--device", type=str, default=ModelArgs.device, help="Device to use (e.g. cuda:0)")
    parser.add_argument("--dtype", type=str, default=ModelArgs.dtype, choices=["float16", "bfloat16"], 
                      help="Floating point precision")
    
    # Checkpointing
    parser.add_argument("--save_checkpoint_dir", type=str, default=ModelArgs.save_checkpoint_dir, 
                      help="Checkpoint save directory")
    parser.add_argument("--save_checkpoint_iter", type=int, default=ModelArgs.save_checkpoint_iter, 
                      help="Checkpoint save interval")
    parser.add_argument("--pre_trained_model_dir", type=str, default=ModelArgs.pre_trained_model_dir, 
                      help="Pretrained model directory")
    # parser.add_argument("--fine_tuned_model_dir", type=str, default=ModelArgs.fine_tuned_model_dir, 
    #                   help="Finetuned model directory")
    
    # Validation/Evaluation
    parser.add_argument("--eval_iters", type=int, default=ModelArgs.eval_iters, help="Evaluation iterations")
    parser.add_argument("--eval_check", type=int, default=ModelArgs.eval_check, help="Evaluation frequency")
    
    # Audio-specific parameters (not configurable via CLI)
    # log_mel_features, kernel_size, stride, sr, SAMPLING_RATE, 
    # N_MELS, WINDOW_DURATION, STRIDE_DURATION, max_t, n_channels
    
    # Experimental features
    parser.add_argument("--use_flash_attention", action='store_true', default=ModelArgs.use_flash_attention,
                      help="Enable flash attention")
    parser.add_argument("--use_liger", action='store_true', default=ModelArgs.use_liger,
                      help="Enable LIGER optimizations")
    parser.add_argument("--use_torch_compile", action='store_true', default=ModelArgs.use_torch_compile,
                      help="Enable torch.compile")
    
    # Additional parameters
    parser.add_argument("--prompt", type=str, default=ModelArgs.prompt, help="Test prompt")
    # parser.add_argument("--train", type=str, required=True, choices=["P"], 
                    #   help="Training mode: P=pretraining/sft")

    args = parser.parse_args()
    return args

def initialize_model_args(args):
    # Create a ModelArgs instance from the parsed arguments
    model_args = ModelArgs(
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        total_batch_size=args.total_batch_size,
        embeddings_dims=args.embeddings_dims,
        no_of_heads=args.no_of_heads,
        no_of_decoder_layers=args.no_of_decoder_layers,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay_optim=args.weight_decay_optim,
        clip=args.clip,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        total_iters=args.total_iters,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        eps=args.eps,
        device=args.device,
        dtype=args.dtype,
        save_checkpoint_dir=args.save_checkpoint_dir,
        save_checkpoint_iter=args.save_checkpoint_iter,
        pre_trained_model_dir=args.pre_trained_model_dir,
        eval_iters=args.eval_iters,
        eval_check=args.eval_check,
        use_flash_attention=args.use_flash_attention,
        use_liger=args.use_liger,
        use_torch_compile=args.use_torch_compile,
        prompt=args.prompt,
        # train=args.train
    )
    return model_args


if __name__ == "__main__":
    args = parse_args()
    

    model_args = initialize_model_args(args)
    
    trainer = Trainer()
    if(args.train == "P"):
        trainer.train()
    elif(args.train == "D"):
        trainer.train_dpo()