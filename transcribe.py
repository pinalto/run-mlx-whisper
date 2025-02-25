import os
import sys
import mlx_whisper

# result = mlx_whisper.transcribe(sys.argv[1],path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="ja")
model = os.path.join(os.path.dirname(__file__), 'model')
result = mlx_whisper.transcribe(sys.argv[1],path_or_hf_repo=model, language="ja")
with open(sys.argv[1] + ".txt", "w") as f:
  for chunk in result["segments"]:
    print("[%.2fs -> %.2fs] %s" % (chunk["start"], chunk["end"], chunk["text"]), file=f)
