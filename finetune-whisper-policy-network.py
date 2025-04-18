import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gc

import whisper
from whisper.streaming import StreamingConfig
from whisper.tokenizer import get_tokenizer
from whisper.training import MonotonicRegularizationLoss

from transformers import get_scheduler

from datasets import load_dataset

dataset_path = "MLCommons/peoples_speech"
config_name = "clean"

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

def load_training_dataset(batch_size=32, filter_func=None):
    train_dataset = load_dataset(dataset_path, config_name, split=train_split, trust_remote_code=True, streaming=True)
    
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
    
    Y = [tokenizer.encode(example['text']) for example in batch]
    
    Y_in = torch.tensor([[[tokenizer.sot] + seq] for seq in Y], device=device)
    Y_out = torch.tensor([[seq + [tokenizer.eot]] for seq in Y], device=device)
    
    return {
        "mels": mels,
        "Y_in": Y_in,
        "Y_out": Y_out,
    }
    
batch_size = 1
#train_data_loader = load_training_dataset(batch_size=batch_size, filter_func=filter_by_length)
train_data_loader = load_training_dataset(batch_size=batch_size, filter_func=lambda x: x)

lr = 1.5e-3 
num_steps = 20

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1
)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=num_steps
)

lambda_latency = 0.01 # the lower the number the more the model will be sensitive to the timing of the audio
lambda_variance = 0.01
monotonic_regularization_loss_fn = MonotonicRegularizationLoss(lambda_latency=lambda_latency, lambda_variance=lambda_variance)

for batch_idx, batch in enumerate(train_data_loader):
    for idx in range(batch_size):
        mel = batch["mels"][idx]
        y_in = batch["Y_in"][idx]
        y_out = batch["Y_out"][idx]
        
        logits, p_choose = model.decoder(y_in, mel, training=True)
        
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)
        
        alphas = [block.alpha for block in model.decoder.blocks]
        
        # average over blocks
        alpha = torch.stack(alphas).mean(0)
        loss += monotonic_regularization_loss_fn(alpha)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        current_lr = lr_scheduler.get_last_lr()[0]
        step = batch_idx * batch_size + idx
        
        beta_weight = min(1.0, step / num_steps)
        for block in model.decoder.blocks:
            block.beta_weight = beta_weight

        print(f"Step {step}: loss = {loss.item():.4f}, lr = {current_lr:.6e}")
        
        del logits, p_choose, alphas, alpha
        gc.collect()
        torch.cuda.empty_cache()

    if batch_idx >= num_steps:
        break
    
save_path = "whisper_streaming_pchoose.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")