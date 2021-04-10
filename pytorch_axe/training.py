
import torch
from torch.nn.utils import clip_grad_norm_

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_dataloader, optimizer, monitor, scheduler=None,
                clip_value=None, device=DEFAULT_DEVICE):
    model.train()
    monitor.reset_epoch()

    for batch in train_dataloader:
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss = model.training_step(batch)
            loss.backward()
            if clip_value is not None:
                clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        monitor.step(loss, batch_size=train_dataloader.batch_size)
    
    monitor.log_epoch("train")
    
def valid_epoch(model, valid_dataloader, optimizer, monitor, loss_fn, 
                device=DEFAULT_DEVICE):
    model.eval()
    monitor.reset_epoch()
    
    for batch in valid_dataloader:
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            loss = model.validation_step(batch)      
        monitor.step(loss, batch_size=valid_dataloader.batch_size)
    
    early_stop = monitor.log_epoch("valid")
    return early_stop
