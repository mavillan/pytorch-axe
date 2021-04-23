import torch
from torch.nn.utils import clip_grad_norm_
from pytorch_axe.monitor import Monitor

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch_to_device(batch, device):
    batch_in_device = [tensor.to(device) for tensor in batch]
    return batch_in_device

def train_epoch(model, train_dataloader, optimizer, monitor, scheduler=None,
                clip_value=None, device=DEFAULT_DEVICE):
    model.train()
    monitor.reset_epoch()

    for batch in train_dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
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
    
def valid_epoch(model, valid_dataloader, optimizer, monitor, 
                device=DEFAULT_DEVICE):
    model.eval()
    monitor.reset_epoch()
    
    for batch in valid_dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(False):
            loss = model.validation_step(batch)      
        monitor.step(loss, batch_size=valid_dataloader.batch_size)
    
    early_stop = monitor.log_epoch("valid")
    return early_stop

def iterative_train(
    model, train_dataloader, valid_dataloader, min_epochs=10, max_epochs=50, 
    patience=10, clip_value=None, metric_fn=None, early_stop_on_metric=False, 
    lower_is_better=True, device=DEFAULT_DEVICE, verbose=True):
    
    # send model to device
    model = model.to(device)
    
    # setup of optimizer and scheduler
    optimizer,scheduler = model.configure_optimizers()
    reduce_on_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    if reduce_on_plateau:
        scheduler_batch_level = None
        scheduler_epoch_level = scheduler

    # setup of monitor
    dataset_sizes = {
        "train":len(train_dataloader.dataset), 
        "valid":len(valid_dataloader.dataset)
        }
    monitor = Monitor(
        model, optimizer, scheduler, patience, metric_fn,
        max_epochs, dataset_sizes, early_stop_on_metric,
        lower_is_better, verbose
        )

    for epoch in monitor.iter_epochs:
        train_epoch(model, train_dataloader, optimizer, monitor, scheduler_batch_level, clip_value, device)    
        early_stop = valid_epoch(model, valid_dataloader, optimizer, monitor, device)
        if early_stop and (epoch-1 >= min_epochs): break
            
        if scheduler_epoch_level is not None and reduce_on_plateau:
            scheduler_epoch_level.step(monitor.epoch_loss["valid"])
        elif scheduler_epoch_level is not None:
            scheduler_epoch_level.step()
            
    return model,monitor

def iterative_predict(model, dataloader, device=DEFAULT_DEVICE):
    model.eval()
    all_preds = list()
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(False):
            pred = model.prediction_step(batch)
            all_preds.append(pred)
    return torch.cat(all_preds)