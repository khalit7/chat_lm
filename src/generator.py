from torch._C import device
from torch.mps import is_available
from .dataset.arabic import get_tokenizer
import torch

class Generator:

    def __init__(self,model) -> None:
        self.model = model
        self.tokenizer = get_tokenizer() 
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"


    def generate(self,prompt="<s>",max_tokens=50,temperature=None,top_p=None,top_k=None):
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt)
            while len(tokens) < max_tokens and tokens[-1] != self.tokenizer.eos_token_id:
                output = self.model(**{"input_ids":torch.tensor(tokens,device=self.device).unsqueeze(0), "attention_mask":torch.ones((1,len(tokens)),device=self.device)})
                token_id = output[-1,:].flatten().argmax().item()
                tokens.append(token_id)


        return "\n".join(self.tokenizer.convert_ids_to_tokens(tokens))

    # TODO: implement kv caching
