"""This module contains a function to record audio from the microphone and save it to a file."""

import sounddevice as sd
import numpy as np
import wave
import threading
from pydub import AudioSegment
import sys
from openai import OpenAI
import os

audio_file_path = "recording"


def record_audio(filename=audio_file_path+".wav", fs=44100, channels=1):
    """ユーザーがエンターキーを押すまで録音を続け、ファイルをOgg Vorbis形式で保存する関数."""
    # 録音用のグローバル変数
    global is_recording
    is_recording = True

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
        """録音されたフレームをWAVファイルに保存する関数."""
        wav_data = np.concatenate(frames, axis=0)
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(np.iinfo(np.int16).bits // 8)
        wav_file.setframerate(fs)
        wav_file.writeframes(np.array(wav_data * 32767, dtype=np.int16))
        wav_file.close()
        print(f"WAVファイル保存: {filename}")

    def convert_to_ogg(wav_filename):
        """WAVファイルをOgg Vorbis形式に変換する関数."""
        ogg_filename = wav_filename.replace(".wav", ".ogg")
        audio = AudioSegment.from_wav(wav_filename)
        audio.export(ogg_filename, format="ogg")
        print(f"Oggファイル変換完了: {ogg_filename}")
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


def convert_speech_to_text(file_path=audio_file_path+".ogg", model="whisper-1", language="ja", temperature=0.0):
    """
    Convert an audio file to text using OpenAI's Whisper API.

    :param audio_file_path: Path to the audio file.
    :param model: The model to use for transcription. Default is 'whisper-large'.
    :param language: Language of the speech. Default is 'Japanese'.
    :return: Transcribed text.
    """
    # Load the audio file
    # with open(file_path, "rb") as audio_file:
    #     audio_data = audio_file.read()
    client = OpenAI()

    # Send the audio data to OpenAI's Whisper API
    transcript = client.audio.transcriptions.create(
        file=open(file_path, "rb"),
        model=model,
        language=language,
        temperature=temperature
    )

    # Extract and return the transcribed text
    return transcript.text


record_audio()
text = convert_speech_to_text()

print(text)
