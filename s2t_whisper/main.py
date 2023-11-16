"""This module contains a function to record audio from the microphone and save it to a file."""

import sounddevice as sd
import numpy as np
import wave
import threading
import sys
from openai import OpenAI

audio_file_path = "recording.wav"


def record_audio(wav_filename=audio_file_path, fs=44100, channels=1):
    """ユーザーがエンターキーを押すまで録音を続け、ファイルに保存する関数."""
    # 録音用のグローバル変数
    global is_recording
    is_recording = True
    wav_filename = "recording.wav"

    def record_internal():
        """内部で使用する録音処理関数."""
        global is_recording
        with sd.InputStream(samplerate=fs, channels=channels) as stream:
            print("録音中... エンターキーを押して停止")
            frames = []
            while is_recording:
                data, overflowed = stream.read(fs)
                frames.append(data)
            return frames

    def save_to_file(frames, filename):
        """録音されたフレームをファイルに保存する関数."""
        # データを連結して保存
        wav_data = np.concatenate(frames, axis=0)
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(np.iinfo(np.int16).bits // 8)
        wav_file.setframerate(fs)
        wav_file.writeframes(np.array(wav_data * 32767, dtype=np.int16))
        wav_file.close()
        print(f"ファイル保存: {filename}")

    # 録音スレッドを開始
    recording_thread = threading.Thread(target=lambda: save_to_file(record_internal(), wav_filename))
    recording_thread.start()

    # エンターキー入力を待機
    input()
    is_recording = False

    # 録音スレッドが終了するのを待つ
    recording_thread.join()


def convert_speech_to_text(file_path=audio_file_path, model="whisper-1", language="ja", temperature=0.0):
    """
    Convert an audio file to text using OpenAI's Whisper API.

    :param audio_file_path: Path to the audio file.
    :param model: The model to use for transcription. Default is 'whisper-large'.
    :param language: Language of the speech. Default is 'Japanese'.
    :return: Transcribed text.
    """
    # Load the audio file
    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()
    client = OpenAI()

    # Send the audio data to OpenAI's Whisper API
    transcript = client.audio.transcriptions.create(
        file=audio_data,
        model=model,
        language=language,
        temperature=temperature
    )

    # Extract and return the transcribed text
    return transcript.text


record_audio()
