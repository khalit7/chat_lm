import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.nn import CrossEntropyLoss
import wandb

from .dataset.arabic import get_tokenizer


class Trainer:
    def __init__(self,model,wandb_run,config) -> None:
        self.model = model
        self.wandb_run = wandb_run
        self.config = config
        tokenizer = get_tokenizer()
        self.loss_fn = CrossEntropyLoss(ignore_index = tokenizer.pad_token_id )
        self.optimizer = Adam(self.model.parameters(),lr=config["learning_rate"])

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"

    def train(self,train_dataloader,val_dataloader):
        self.model.to(self.device)
        self.model.train()
        warmup_steps = len(train_dataloader)*self.config["warmup_percentage"]
        linear_lr_scheduler = LinearLR(self.optimizer,start_factor=self.config["warmup_start_factor"],end_factor=1,total_iters=warmup_steps)
        cosine_lr_scheduler = CosineAnnealingLR(self.optimizer,T_max=len(train_dataloader)-warmup_steps)
        self.lr_scheduler = SequentialLR(self.optimizer,[linear_lr_scheduler,cosine_lr_scheduler],milestones=[warmup_steps])
        print("Starting training on device = ",self.device)

        for epoch in range(self.config["num_epochs"]):
            for step,(X,Y) in tqdm(enumerate(train_dataloader)):

                # run evaluation
                if step%self.config["eval_every"] == 0:
                    self.eval(val_dataloader,step)

                # run training step
                self.train_step(X,Y,step)



    def train_step(self,X,Y,step):
        X = { k:v.to(self.device) for k,v in X.items() }
        Y = Y.to(self.device)
        output = self.model(**X)
        loss = self.loss_fn(output,Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.wandb_run.log(
                {
                    "train_loss":loss.detach().item(),
                    "learning_rate":self.optimizer.param_groups[0]["lr"]
                },
                step=step
                )


    def eval(self,dataloader,step):
        self.model.eval()



        self.model.train()
