from datasets import load_dataset
import json
from tqdm import tqdm

import torch

import sys
import numpy as np
import librosa
import logging
import torch

from whisper_streaming.whisper_online import asr_factory as whisper_streaming_factory
from whisper.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(audio, beg, end):
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

def set_logging(args,logger,other="_server"):
    logging.basicConfig(#format='%(name)s 
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online"+other).setLevel(args.log_level)
#    logging.getLogger("whisper_online_server").setLevel(args.log_level)

# Simulate a streaming audio source
def chunk_audio(audio_tensor, chunk_size):
    for i in range(0, len(audio_tensor), chunk_size):
        yield audio_tensor[i:i+chunk_size]

# Whisper backend
def add_whisper_streaming_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--model', type=str, default='tiny.en', help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="whisper", choices=["faster-whisper", "whisper_timestamped", "whisper", "mlx-whisper", "openai-api"],help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
    parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level", default='INFO')

class Runner:
    def __init__(self) -> None:
        self.asr = None
        self.online = None
        self.audio = None
        self.args = None
        self.duration = 0

    def init(self, args):
        self.args = args
        self.audio = load_audio(args.audio_path)

        duration = len(self.audio)/SAMPLE_RATE
        logger.debug("Audio duration is: %2.2f seconds" % duration)

        set_logging(args,logger)

        if self.asr is None:
            self.asr, self.online = whisper_streaming_factory(args)

    def run(self):
        chunk_size = 16000  # 1 second chunks at 16kHz
        stream = chunk_audio(self.audio, chunk_size)

        if self.audio is None or self.asr is None:
            print("Uninitialized")
            return

        self.online.init()

        chunks = []
        for a in stream:
            a = self.online.insert_audio_chunk(a)
            o = self.online.process_iter()
            chunks.append(o[2])

        o = self.online.finish()
        chunks.append(o[2])

        return chunks

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--audio_path', default="./jfk.wav", type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
add_whisper_streaming_args(parser)
parser.add_argument("--method", choices=['whisper_streaming', 'seamless', 'offline'], help="The streaming method to use", default='seamless')

args = parser.parse_args()

# reset to store stderr to different file stream, e.g. open(os.devnull,"w")
logfile = sys.stderr

runner = Runner()


# Load a single example from the streaming dataset
dataset = load_dataset("MLCommons/peoples_speech", "clean", split="test", streaming=True, trust_remote_code=True)

# You can apply filtering logic here if needed
dataset = dataset.filter(lambda example: example["duration_ms"] > 5000)


# Warmup
print("Warming up...")
runner.init(args)
runner.run()
print("Done")


batch = []

NUM_EXAMPLES = 200

for i, example in enumerate(tqdm(dataset, total=NUM_EXAMPLES)):
    if i >= NUM_EXAMPLES:
        break
    audio_array = example["audio"]["array"].astype(np.float32)
    text_reference = example["text"]

    try:
        runner.init(args)
        runner.audio = audio_array
        o = runner.run()
    except Exception as e:
        print(i, example["id"], example["duration_ms"], e)
        o = []

    result = {
        "text": "".join(o).strip(),
        "delays":  [[word, float(j + 1)] for j, s in enumerate(o) for word in s.split()]
    }
    batch.append(result)

    if (i+1) % 25 == 0:
        with open(f"./results/whisper_streaming-batch-{i+1}.json", "w") as f:
            json.dump(batch, f)
        batch.clear()
