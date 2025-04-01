import whisper

w = whisper.load_model("/home/luv/.cache/whisper/tiny.en.pt")
print(w.transcribe('./jfk.wav'))

m = whisper.load_streaming_model("/home/luv/.cache/whisper/stream-tiny.en.pt", strict=False)
print(m.transcribe_stream('./jfk.wav', chunk_size=32000))
