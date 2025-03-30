from config import ModelArgs
from model import Llama
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
import argparse


tokenizer = Tokenizer()
tokenizer = tokenizer.ready_tokenizer()


def preprocess_audio(file_path):
    spectrogram = librosa.feature.melspectrogram(
        y=librosa.load(file_path, sr=ModelArgs.SAMPLING_RATE)[0],
        sr=ModelArgs.SAMPLING_RATE,
        n_mels=ModelArgs.N_MELS,
        n_fft=ModelArgs.N_FFT,
        hop_length=ModelArgs.HOP_LENGTH,
        win_length=ModelArgs.WIN_LENGTH,
        fmax=ModelArgs.FMAX
    )
    return spectrogram

# test_iter = iter(test_loader)
def topk_sampling(model, audio_path, max_length=30, top_k=50, temperature=1.0, device='cuda'):
    # Get test batch (batch_size=1)
   


    # Extract inputs
    spectrogram = preprocess_audio(audio_path)
    spectrogram = spectrogram.to(device)
    # spectrogram = batch['spectrogram'].to(device)  # [1, n_mels, time]
    input_text = "<|startoftranscript|>en<|transcribe|>"  # Initial prompt
    global block_size
    # Tokenize initial input
    input_ids = tokenizer(
        input_text,
        return_tensors='pt',
        # max_length=block_size,
        # truncation=True,
        # padding='max_length'
    ).input_ids.to(device)
    len_input_id = len(input_ids)
    model.eval()
    generated_ids = input_ids.clone()
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Generation loop
        # print("Now: ", len(generated_ids))
        for _ in range(max_length):
            # Forward pass through full model
            outputs = model(
                src=spectrogram,  # Spectrogram input
                tgt=generated_ids,  # Text tokens
                src_mask=None,
                tgt_mask=None,
                actual_labels=None,
                inference=True
            )
            
            # Get last token logits
            logits = outputs[:, -1, :]
            
            # Apply temperature scaling
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            # Top-k filtering
            probs, indices = torch.topk(probs, top_k)
            # indices = indices
            
            
            # Sample from top-k
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, next_token)
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            # print("Generated ids: ", generated_ids.shape)
            # print(generated_ids)
            # if(generated_ids.shape[1] >= block_size):
            #     break
            # Stop if EOT generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode and clean
    transcript = tokenizer.decode(
        generated_ids[0], 
        skip_special_tokens=True
    )
    # real_sentence = batch['real_text'][0]
    
    return transcript


def main():

    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="Enter audio path")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model_path", type=str, default="Enter model path")
    # parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()
    
    model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
    # model = torch.compile(model)
    model = model.to(ModelArgs.device)

    dict_model = torch.load(args.model_path)
    dict_model['MODEL_STATE'] = remove_prefix(dict_model['MODEL_STATE'], '_orig_mod.')
    model.load_state_dict(dict_model['MODEL_STATE'])
    model.eval()
    print("Model ready")
    # prompt = 'Its a secret'

    with torch.no_grad():
        generated_text = topk_sampling(model, args.audio_path, max_length=args.max_length, top_k=50, temperature=args.temperature, device=ModelArgs.device)
        print("Generated: ", generated_text)



if __name__ == '__main__':
    main()
