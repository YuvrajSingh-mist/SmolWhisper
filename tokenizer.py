
from transformers import AutoTokenizer
import os


class Tokenizer:
    
    def __init__(self) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = '...')

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        
        SOT = '<|startoftranscript|>'
        EOT = '<|endoftranscript|>'
        transcribe = '<|transcribe|>'
        prev = '<|prev|>'

        special_tokens_dict = {
            'additional_special_tokens': [SOT, EOT, transcribe, prev]
        }


        self.tokenizer.add_special_tokens(special_tokens_dict)

    def ready_tokenizer(self):
        
        return self.tokenizer
    
    



