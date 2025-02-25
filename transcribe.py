import os
import sys
import argparse
import mlx_whisper

def transcribe_audio_segment(input_file, start_time=None, end_time=None):
    """
    指定された時刻範囲でオーディオファイルをトランスクリプションする
    
    Args:
        input_file: 入力オーディオファイルのパス
        start_time: 開始時刻（秒）、省略時はファイルの最初から
        end_time: 終了時刻（秒）、省略時はファイルの最後まで
    
    Returns:
        トランスクリプション結果の辞書
    """
    # ファイルが存在するか確認
    if not os.path.exists(input_file):
        print(f"エラー: ファイル '{input_file}' が見つかりません。")
        sys.exit(1)
    
    # 開始時刻と終了時刻のバリデーションと調整
    if start_time is not None and start_time < 0:
        print("警告: 開始時刻が負の値です。0秒に設定します。")
        start_time = 0
    
    if end_time is not None and start_time is not None and end_time <= start_time:
        print("エラー: 終了時刻は開始時刻より大きい必要があります。")
        sys.exit(1)
    
    
    # トランスクリプション実行
    model = os.path.join(os.path.dirname(__file__), 'model')

    if end_time is not None:
      return mlx_whisper.transcribe(
          input_file,
          path_or_hf_repo=model,
          clip_timestamps=[start_time,end_time]
      )
    elif start_time is not None:
      return mlx_whisper.transcribe(
          input_file,
          path_or_hf_repo=model,
          clip_timestamps=[start_time]
      )
    else:
      return mlx_whisper.transcribe(
          input_file,
          path_or_hf_repo=model,
        )

def main():
    parser = argparse.ArgumentParser(description="MLX Whisperを使って音声ファイルの指定された部分をトランスクリプションします。")
    parser.add_argument("input_file", help="入力オーディオファイルのパス")
    parser.add_argument("start_time", nargs="?", type=float, default=None, help="開始時刻（秒）、省略時はファイルの最初から")
    parser.add_argument("end_time", nargs="?", type=float, default=None, help="終了時刻（秒）、省略時はファイルの最後まで")
    
    args = parser.parse_args()
    
    # トランスクリプション実行
    result = transcribe_audio_segment(args.input_file, args.start_time, args.end_time)
    
    # 結果を出力
    with open(args.input_file + ".txt", "w") as f:
        if "segments" in result and result["segments"]:
            for i, segment in enumerate(result["segments"]):
                text = segment.get("text", "")
                if text:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    print(f"[{start:.2f}s - {end:.2f}s] {text}", file=f)
        else:
            print(result["text"], file=f)


if __name__ == "__main__":
    main()
