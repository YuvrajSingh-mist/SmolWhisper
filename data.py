
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from config import ModelArgs



tokenizer = Tokenizer().ready_tokenizer()


import argparse
import numpy as np

HF_TOKEN = '...'
gs = load_dataset("speechcolab/gigaspeech", "s", token=HF_TOKEN, trust_remote_code=True)



# gs = gs['train'].train_test_split(shuffle=True,test_size=0.5)
validation_data = gs['validation']
combined = concatenate_datasets([gs['train'], gs['test']])

# Next, split the combined dataset so that the validation and test sets are about 1,000 each.
# We'll first split out 2,000 samples for validation+test.
split_result = combined.train_test_split(test_size=2000, shuffle=True, seed=42)
train_data = split_result['train']


# Now, split the 2,000 samples into two equal halves: 1,000 for validation and 1,000 for test.
split_temp = validation_data.train_test_split(test_size=0.5, shuffle=True, seed=42)
test_data = split_temp['test']




MAX_DURATION_IN_SECONDS = 30

import librosa
from tqdm import tqdm
def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS

train_new_column = []

for x in tqdm(range(len(train_data))):
    train_new_column.append(librosa.get_duration(path=train_data[x]['audio']['path']))

gs_ = train_data.add_column("duration", train_new_column)


gs_ = gs_.filter(is_audio_length_in_range, input_columns=["duration"])


truncated_gs_train = gs_.remove_columns(["duration"])
# truncated_gs



val_new_column = []
# new_column = [librosa.get_duration(path=x) ]]
for x in tqdm(range(len(validation_data))):
    val_new_column.append(librosa.get_duration(path=validation_data[x]['audio']['path']))

gs_ = validation_data.add_column("duration", val_new_column)


gs_ = gs_.filter(is_audio_length_in_range, input_columns=["duration"])


truncated_gs_val = gs_.remove_columns(["duration"])
# truncated_gs

test_new_column = []
# new_column = [librosa.get_duration(path=x) ]]
for x in tqdm(range(len(test_data))):
    test_new_column.append(librosa.get_duration(path=test_data[x]['audio']['path']))

gs_ = test_data.add_column("duration", test_new_column)


gs_ = gs_.filter(is_audio_length_in_range, input_columns=["duration"])


truncated_gs_test = gs_.remove_columns(["duration"])





def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)

    def collate_fn(batch):

        # MAX_FRAMES = int(MAX_DURATION_IN_SECONDS / STRIDE_DURATION)

        def pad_to_max_t(spectrogram, max_t):

            n_mels, t = spectrogram.shape
            if t < max_t:
                # Pad with zeros
                pad_width = ((0, 0), (0, max_t - t))
                spectrogram = np.pad(spectrogram, pad_width, mode='constant')
            else:
                spectrogram = spectrogram[:, :max_t]

            return spectrogram

        def clean(desc):
            # Use regex to remove anything between < and >
            cleaned_text = re.sub(r'<[^>]*>', '', desc)
            return cleaned_text

        # Audio processing parameters
        n_fft = int(ModelArgs.SAMPLING_RATE * ModelArgs.WINDOW_DURATION)
        hop_length = int(ModelArgs.SAMPLING_RATE * ModelArgs.STRIDE_DURATION)
        
        batch_spectrograms = []
        batch_input_ids = []
        batch_text = []
        batch_labels = []
        
        for item in batch:


            spectrogram = librosa.feature.melspectrogram(
                y=item['audio']['array'],
                sr=ModelArgs.SAMPLING_RATE,
                n_mels=ModelArgs.N_MELS,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                fmax=ModelArgs.SAMPLING_RATE // 2
            )
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            SOT = '<|startoftranscript|>'
            EOT = '<|endoftranscript|>'
            transcribe = '<|transcribe|>'
            # prev = '<|prev|>'
            spectrogram = pad_to_max_t(spectrogram, block_size)
            # probs = round(random.random(),1)
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

            # if(probs == 0.5):
                # Normalize the spectrogram between -1 and 1
            spectrogram_min = spectrogram.min()
            spectrogram_max = spectrogram.max()
            # spectrogram = spectrogram.unsqueeze(0)  # Shape: (1, n_mels, max_t)
            # prev_text =
            text = clean(item['text'])
            original_text = text.lower()
            text = text.lower()
            text = SOT  + 'en' + transcribe +  text + EOT
            tokenized_text = tokenizer(text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt')
            # print(tokenized_text.shape)

            epsilon = 1e-8  # To avoid division by zero
            spectrogram = 2 * ((spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + epsilon)) - 1

            # tokenized_win_prompt = tokenizer(text, max_length = ModelArgs.block_size, padding='max_length', truncation=True,  return_tensors="pt").to(device)
            tokenized_text['labels'] = tokenized_text['input_ids'].clone()
            tokenized_text['labels'][: , :-1] = tokenized_text['input_ids'][: , 1:]
            tokenized_text['labels'][: , -1] = tokenizer.eos_token_id

            tokenized_text_x = tokenized_text['input_ids'].squeeze(0)
            tokenized_text_y = tokenized_text['labels'].squeeze(0)

            batch_spectrograms.append(spectrogram)
            batch_input_ids.append(tokenized_text_x)
            batch_labels.append(tokenized_text_y)
            batch_text.append(original_text)
        return {
            "real_text": batch_text,
            'spectrogram': torch.stack(batch_spectrograms),
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels)
        }

    
    dataloader = None

    # if(tinystories):
    if(split == 'train'):
            data_loader = DataLoader(
            truncated_gs_train,
            # generator=generator,
            batch_size=batch_size,
             
            sampler=DistributedSampler(truncated_gs_train, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True
        )
    elif(split == 'val'):
            data_loader = DataLoader(
            truncated_gs_val,

            batch_size=batch_size,
            sampler=DistributedSampler(truncated_gs_val, shuffle=False),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True
        )

    elif(split == 'test'):
        data_loader = DataLoader(
            truncated_gs_test,
            batch_size=1,
            sampler=DistributedSampler( truncated_gs_test, shuffle=False),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True
        )
    
    return data_loader

