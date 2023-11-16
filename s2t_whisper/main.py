"""This module contains a function to record audio from the microphone and save it to a file."""

import argparse
import os
import sys
import threading
import time
import wave

import numpy as np
import pyperclip
import sounddevice as sd
from openai import OpenAI
from pydub import AudioSegment

audio_file_path = "recording"
colors = {
    "red": "\033[91m",
    "grey": "\033[90m",
    "end": "\033[0m",
}


def print_colored(text, color):
    """指定された色でテキストを表示する関数."""
    color_code = colors.get(color, colors["end"])
    print(f"{color_code}{text}{colors['end']}")


def record_audio(filename=audio_file_path+".wav", fs=44100, channels=1):
    """ユーザーがエンターキーを押すまで録音を続け、ファイルをOgg Vorbis形式で保存する関数."""
    # 録音用のグローバル変数
    global is_recording
    is_recording = True

    def record_internal():
        """内部で使用する録音処理関数."""
        global is_recording
        start_time = time.time()
        with sd.InputStream(samplerate=fs, channels=channels) as stream:
            frames = []
            while is_recording:
                data, _ = stream.read(fs)
                frames.append(data)

                # 経過時間の表示（秒単位でカウントアップ）
                elapsed_time = int(time.time() - start_time)
                sys.stdout.write(f"\r{colors['red']}Recording in progress: {colors['end']}{elapsed_time} sec {colors['grey']}Press \"Enter\" to stop{colors['end']}")  # noqa: E501
                sys.stdout.flush()

            time.sleep(1)
            return frames

    def save_to_file(frames, filename):
        """録音されたフレームをWAVファイルに保存する関数."""
        wav_data = np.concatenate(frames, axis=0)
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(np.iinfo(np.int16).bits // 8)
        wav_file.setframerate(fs)
        wav_file.writeframes(np.array(wav_data * 32767, dtype=np.int16))
        wav_file.close()
        print("\n")
        print_colored("Save WAV files...", "grey")

    def convert_to_ogg(wav_filename):
        """WAVファイルをOgg Vorbis形式に変換する関数."""
        ogg_filename = wav_filename.replace(".wav", ".ogg")
        audio = AudioSegment.from_wav(wav_filename)
        audio.export(ogg_filename, format="ogg")
        print_colored("Convert to Ogg file...", "grey")
        os.remove(wav_filename)  # 元のWAVファイルを削除
        return ogg_filename

    # 録音スレッドを開始
    recording_thread = threading.Thread(target=lambda: save_to_file(record_internal(), filename))
    recording_thread.start()

    # エンターキー入力を待機
    input()
    is_recording = False

    # 録音スレッドが終了するのを待つ
    recording_thread.join()

    # WAVファイルをOgg Vorbis形式に変換
    convert_to_ogg(filename)


def convert_speech_to_text(file_path=audio_file_path+".ogg", model="whisper-1", language="ja", temperature=0.0, prompt=""):
    """Convert an audio file to text using OpenAI's Whisper API."""
    print_colored("Convert to text...", "grey")
    print_colored(f" - model: {model}", "grey")
    print_colored(f" - lang.: {language}", "grey")
    print_colored(f" - temp.: {temperature}", "grey")
    print_colored(f" - prompt: {prompt[:45]}...", "grey")
    client = OpenAI()
    try:
        # Send the audio data to OpenAI's Whisper API
        transcript = client.audio.transcriptions.create(
            file=open(file_path, "rb"),
            model=model,
            language=language,
            temperature=temperature,
            prompt=prompt
        )
    except Exception as e:
        print(f"Error: {e}")
        raise e

    # Extract and return the transcribed text
    return transcript.text


def print_and_copy(text) -> None:
    """ターミナルに表示し、クリップボードにコピーする関数."""
    print('\n')
    print(text)
    print('\n')
    pyperclip.copy(text)


def main(*, model, language, temperature, prompt) -> str:
    """メイン関数."""
    # 録音
    record_audio()

    # 音声認識
    try:
        text = convert_speech_to_text(model=model, language=language, temperature=temperature, prompt=prompt)
    except Exception:
        text = convert_speech_to_text(model=model, language=language, temperature=temperature, prompt=prompt)

    return text


def app_run(*, model, language, temperature, prompt):
    """アプリケーションを実行する関数."""
    while True:
        # ユーザー入力を受け取る
        print_colored('Press "enter" to start recording, press "q" to exit: ', "grey")
        user_input = input()

        # 'q'または'Q'が入力されたら終了
        if user_input.lower() == 'q':
            print_colored("Exit the program.", "grey")
            break

        text = main(**vars(parser.parse_args()))

        # 認識結果を表示
        print_and_copy(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This program records audio from the microphone and saves it to a file.
        """
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="whisper-1",
        choices=["whisper-1"],
        help="The model to use for transcription. Default is 'whisper-1'.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="ja",
        help="Language of the speech. Default is 'Japanese'.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature of the speech. 0.0 to 1.0.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="I am in charge of planning, development (front-end and back-end), and operation all by myself. The front-end development is component-oriented using Next.js, while the back-end uses Python as the main language and Django as a library. I like object-oriented programming and work on development following domain-driven design. ",  # noqa: E501
        help="Prompt for the speech.",
    )

    app_run(**vars(parser.parse_args()))
