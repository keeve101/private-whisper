import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import whisper
from whisper.streaming import StreamingConfig
from whisper.tokenizer import get_tokenizer
from whisper.training import MonotonicRegularizationLoss

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
        mels = [model.encoder(mel.unsqueeze(0)) for mel in mels]
    
    Y = [tokenizer.encode(example['text']) for example in batch]
    
    Y_in = torch.tensor([[[tokenizer.sot] + seq] for seq in Y], device=device)
    Y_out = torch.tensor([[seq + [tokenizer.eot]] for seq in Y], device=device)
    
    return {
        "mels": mels,
        "Y_in": Y_in,
        "Y_out": Y_out,
    }
    
batch_size = 1
train_data_loader = load_training_dataset(batch_size=batch_size, filter_func=filter_by_length)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
monotonic_regularization_loss_fn = MonotonicRegularizationLoss()

mundane = 10

for batch_idx, batch in enumerate(train_data_loader):
    for idx in range(batch_size):
        mel = batch["mels"][idx]
        y_in = batch["Y_in"][idx]
        y_out = batch["Y_out"][idx]
        
        logits, p_choose = model.decoder(y_in, mel, training=False)
        
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)
        
        alphas = []
        for block in model.decoder.blocks:
            alphas.append(block.alpha)
        
        # average over blocks
        alpha = torch.stack(alphas).mean(0)
        loss += monotonic_regularization_loss_fn(alpha)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss)
    if batch_idx == mundane:
        break