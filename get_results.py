import whisper
import json
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer
from whisper.normalizers import EnglishTextNormalizer
import numpy as np

import evaluate

from datasets import load_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load streaming dataset
dataset = load_dataset("MLCommons/peoples_speech", "clean", split="test", streaming=True, trust_remote_code=True)
dataset = dataset.filter(lambda example: example["duration_ms"] > 5000)


# Load models and tokenizer
whisper_tiny = whisper.load_model("tiny.en").to(device)
tokenizer = get_tokenizer(multilingual=False)

def load_results(configs):
    d = {}
    for k,v in configs.items():
        d[k] = []
        for batch in ['25', '50', '75', '100', '125', '150', '175', '200']:
            with open(f'./results/{v}-batch-{batch}.json') as f:
                d[k].extend(json.load(f))
    return d

def process_example(example, idx, results, configs, final_results):
    normalizer = EnglishTextNormalizer()
    audio_array = torch.tensor(example["audio"]["array"], device=device)

    result = whisper_tiny.transcribe(audio_array.to(torch.float32), word_timestamps=True)
    whisper_pred = result["text"].strip()
    final_results['text_refs'].append(normalizer(example['text']))
    final_results['pred']['whisper'].append(normalizer(whisper_pred))
    token_audio_times = []
    
    for segment in result["segments"]:
        for word_info in segment["words"]:
            token_audio_times.extend([word_info["start"]])

    for k in configs: 
        # Compute final transcript and average lag
        r = results[k][idx]
        all_words = r['text'].split()
        token_delays = r['delays']

        word_idx_to_token_idx = {}
        token_idx = 0

        for word_idx, word in enumerate(all_words):
            word_idx_to_token_idx[word_idx] = []

            current_subword = ""
            while token_idx < len(token_delays):
                decoded_token = token_delays[token_idx][0]

                if decoded_token == "":
                    token_idx += 1
                    continue

                current_subword += decoded_token
                word_idx_to_token_idx[word_idx].append(token_idx)
                token_idx += 1

                if current_subword == word:
                    break

        word_emission_times = {}

        for word_idx, token_indices in word_idx_to_token_idx.items():
            try:
                timestamps = [token_delays[i][1] for i in token_indices if i < len(token_delays)]
                if timestamps:
                    word_emission_times[word_idx] = min(timestamps)
            except IndexError:
                continue

        min_len = min(len(word_emission_times), len(token_audio_times))

        lags = [
            token_delays[i][1] - token_audio_times[i]
            for i in range(min_len)
        ]

        # Drop upper quartile (keep 25th to 75th percentile)
        lags_sorted = sorted(lags)
        cut_lags = lags_sorted[len(lags)//4 : 3*len(lags)//4]
        average_lagging = sum(cut_lags) / len(cut_lags) if cut_lags else float('nan')
        
        final_results['pred'][k].append(normalizer(r['text']))
        final_results['average_lagging'][k].append(average_lagging)
        
    return final_results

# ==== MAIN LOOP =====
num_examples = 200

configs = {
    'emma-pchoose_only': 'our',
    'emma-cross_attn-0.15': 'our-900-0.15',
    'emma-cross_attn-0.17': 'our-900-0.17',
    'emma-cross_attn-0.20': 'our-900-0.2',
    'emma-cross_attn-0.40': 'our-900-0.4',
    'whisper_streaming': 'whisper_streaming',
}

final_results = {
    'text_refs': [],
    'pred': {
        k: [] for k in configs.keys()
    },
    'average_lagging': {
        k: [] for k in configs.keys()
    }
}

final_results['pred']['whisper'] = []


sacrebleu_metric = evaluate.load("sacrebleu")
wer_metric = evaluate.load("wer")

results = load_results(configs)

for i, example in enumerate(tqdm(dataset, total=num_examples)):
    if i >= num_examples:
        break

    r = process_example(example, i, results, configs, final_results)

with open('./results/final.json', 'w') as f:
    json.dump(final_results, f)
# with open('./results/final.json') as f:
#     final_results = json.load(f)


def print_metrics(final_results, config):
    print(f"Results for {config}:")

    bleu = sacrebleu_metric.compute(predictions=final_results['pred'][config], references=final_results['text_refs'])
    print(f"  sacreBLEU: {bleu['score']:.3f}")

    wer = wer_metric.compute(predictions=final_results['pred'][config], references=final_results['text_refs'])
    print(f"        WER: {wer:.3f}")
    if config in final_results['average_lagging']:
        avg_lag = np.nanmean(final_results['average_lagging'][config])
        print(f"Avg Lagging: {avg_lag:.3f}")

print_metrics(final_results, 'whisper')
for config in configs:
    print_metrics(final_results, config)
