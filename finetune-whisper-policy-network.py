import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gc
import json

import whisper
from whisper.streaming import StreamingConfig
from whisper.tokenizer import get_tokenizer
from whisper.training import MonotonicRegularizationLoss

from transformers import get_scheduler

from datasets import load_dataset

dataset_path = "mozilla-foundation/common_voice_17_0"
config_name = "en"

train_split = "train"
validation_split = "validation"
test_split = "test"

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_tiny = whisper.load_model("tiny.en").to(device)
model = whisper.WhisperStreaming(whisper_tiny.dims, StreamingConfig()).to(device)
model.load_state_dict(whisper_tiny.state_dict(), strict=False)

del whisper_tiny

for name, param in model.named_parameters():
    if "p_choose_layer" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        
# for my sanity
for name, param in model.named_parameters():
    if "p_choose_layer" in name:
        assert param.requires_grad == True
    else:
        assert param.requires_grad == False

tokenizer = get_tokenizer(multilingual=False)

def load_dataset_split(batch_size=32, filter_func=None, split="train"):
    train_dataset = load_dataset(dataset_path, config_name, split=split, trust_remote_code=True, streaming=True)
    
    train_dataset = train_dataset.shuffle(seed=0)
    
    if filter_func:
        train_dataset = train_dataset.filter(filter_func)

    data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return data_loader

def filter_by_length(example, min_length=10000):
    audio_length = example["duration_ms"]
    
    return audio_length >= min_length

def collate_fn(batch):
    audio_batch = [whisper.pad_or_trim(torch.tensor(example["audio"]["array"], device=device)) for example in batch]
    mels = [whisper.log_mel_spectrogram(audio.to(torch.float32), n_mels=model.dims.n_mels).to(device) for audio in audio_batch]
    
    with torch.no_grad():
        mels = [model.encoder(mel.unsqueeze(0).detach()) for mel in mels]
    
    Y = [tokenizer.encode(example['sentence'].strip()) for example in batch]
    
    Y_in = torch.tensor([[[tokenizer.sot] + seq] for seq in Y], device=device)
    Y_out = torch.tensor([[seq + [tokenizer.eot]] for seq in Y], device=device)
    
    return {
        "mels": mels,
        "Y_in": Y_in,
        "Y_out": Y_out,
    }
    
train_loss_log = []
train_ce_loss_log = []
train_reg_loss_log = []

val_loss_log = {}
val_ce_loss_log = {}
val_reg_loss_log = {}
    
batch_size = 1
#train_data_loader = load_training_dataset(batch_size=batch_size, filter_func=filter_by_length)
train_data_loader = load_dataset_split(batch_size=batch_size, filter_func=lambda x: x, split=train_split)
validation_data_loader = load_dataset_split(batch_size=batch_size, filter_func=lambda x: x, split=validation_split)

lr = 1.5e-3 / 40
num_steps = 300

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1
)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=num_steps
)

lambda_latency = 0.5 # the lower the number the more the model will be sensitive to the timing of the audio
lambda_variance = 0.5
monotonic_regularization_loss_fn = MonotonicRegularizationLoss(lambda_latency=lambda_latency, lambda_variance=lambda_variance)

num_validation_steps = 20

for batch_idx, batch in enumerate(train_data_loader):
    for idx in range(batch_size):
        mel = batch["mels"][idx]
        y_in = batch["Y_in"][idx]
        y_out = batch["Y_out"][idx]
        
        logits, p_choose = model.decoder(y_in, mel, training=True)
        
        ce_loss = F.cross_entropy(logits.transpose(1, 2), y_out)
        
        alphas = [block.alpha for block in model.decoder.blocks]
        
        # average over blocks
        alpha = torch.stack(alphas).mean(0)
        reg_loss = monotonic_regularization_loss_fn(alpha)
        
        loss = ce_loss + reg_loss
        loss.backward()
        
        # for logging losses
        train_loss_log.append(loss.item())
        train_ce_loss_log.append(ce_loss.item())
        train_reg_loss_log.append(reg_loss.item())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        current_lr = lr_scheduler.get_last_lr()[0]
        step = batch_idx * batch_size + idx + 1
        print(f"Step {step}: total = {loss.item():.4f}, CE = {ce_loss.item():.4f}, Reg = {reg_loss.item():.4f}, lr = {current_lr:.6e}")
        
        next_token = logits[:, -1].argmax(dim=-1).item()
        print(tokenizer.decode(y_in.view(-1).tolist() + [next_token]))
        
        del logits, p_choose, alphas, alpha
        gc.collect()
        torch.cuda.empty_cache()
        
    if step % 40 == 0:
        import os
        os.makedirs("output_3", exist_ok=True)
        save_path = "output_3" +  f"/whisper_streaming_pchoose_{step}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Validation loop
        print("Running validation...")
        val_losses = []
        val_reg_losses = []

        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(validation_data_loader):
                for val_idx in range(batch_size):
                    mel = val_batch["mels"][val_idx]
                    y_in = val_batch["Y_in"][val_idx]
                    y_out = val_batch["Y_out"][val_idx]

                    logits, p_choose = model.decoder(y_in, mel, training=True)

                    ce_loss = F.cross_entropy(logits.transpose(1, 2), y_out)

                    alphas = [block.alpha for block in model.decoder.blocks]
                    alpha = torch.stack(alphas).mean(0)
                    reg_loss = monotonic_regularization_loss_fn(alpha)

                    loss = ce_loss + reg_loss
                    val_losses.append(loss.item())
                    val_reg_losses.append(reg_loss.item())
                    
                    val_loss_log.setdefault(step, [])
                    val_loss_log[step].append(loss.item())
                    
                    val_ce_loss_log.setdefault(step, [])
                    val_ce_loss_log[step].append(ce_loss.item())

                    val_reg_loss_log.setdefault(step, [])
                    val_reg_loss_log[step].append(reg_loss.item())

                    next_token = logits[:, -1].argmax(dim=-1).item()
                    print("Val prediction:", tokenizer.decode(y_in.view(-1).tolist() + [next_token]))

                    del logits, p_choose, alphas, alpha
                    gc.collect()
                    torch.cuda.empty_cache()

                # Only do a few batches
                if val_batch_idx + 1 >= num_validation_steps:
                    break

        print(f"Validation Loss: total = {sum(val_losses)/len(val_losses):.4f}, Reg = {sum(val_reg_losses)/len(val_reg_losses):.4f}")

    if batch_idx >= num_steps:
        break
    
log_path = "output_3/loss_logs.json"
log_data = {
    "step": step,
    "train_loss": train_loss_log,
    "train_ce_loss": train_ce_loss_log,
    "train_reg_loss": train_reg_loss_log,
    "val_loss": val_loss_log,
    "val_ce_loss": val_ce_loss_log,
    "val_reg_loss": val_reg_loss_log
}

with open(log_path, "w") as f:
    json.dump(log_data, f, indent=2)

print(f"Loss logs saved to {log_path}")
