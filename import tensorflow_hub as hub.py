import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import sounddevice as sd
import librosa
import pygame
import queue
import time
import urllib.request
import csv

# ==== 参数配置 ====
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.25  # 将处理块缩短为 0.25 秒，减少响应延迟
CHANNELS = 1
THRESHOLD = 0.2  # 识别分数阈值
COOLDOWN = 1.0   # 音效播放冷却时间（单位：秒）
BUFFER = queue.Queue()
last_played = {}  # 记录每类声音的上次播放时间

# ==== 音频播放初始化 ====
pygame.mixer.init()
sound_map = {
    "Speech": "sounds/speech_pad.wav",
    "Car horn": "sounds/car_drums.wav",
    "Traffic noise": "sounds/traffic_rhythm.wav",
    "Beep": "sounds/beep_click.wav"
}
loaded_sounds = {label: pygame.mixer.Sound(path) for label, path in sound_map.items()}

# ==== 加载 YAMNet 模型 ====
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# ==== 加载标签 ====
LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
with urllib.request.urlopen(LABELS_URL) as f:
    reader = csv.DictReader(f.read().decode('utf-8').splitlines())
    class_names = [row['display_name'] for row in reader]

# ==== 录音回调函数 ====
def audio_callback(indata, frames, time_info, status):
    BUFFER.put(indata.copy())

# ==== 启动麦克风监听 ====
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
                        callback=audio_callback)
stream.start()

# ==== 主识别与反馈循环 ====
print("🎧 实时音频监听中，按 Ctrl+C 停止。")

try:
    while True:
        if not BUFFER.empty():
            audio_block = BUFFER.get()
            waveform = np.reshape(audio_block, (-1,)).astype(np.float32)

            # 模型预测
            scores, embeddings, spectrogram = yamnet_model(waveform)
            mean_scores = np.mean(scores.numpy(), axis=0)
            top_indices = np.argwhere(mean_scores > THRESHOLD).flatten()

            # 音轨播放逻辑
            now = time.time()
            for idx in top_indices:
                label = class_names[idx]
                for key in sound_map:
                    if key.lower() in label.lower():
                        if (key not in last_played) or (now - last_played[key] > COOLDOWN):
                            loaded_sounds[key].play()
                            print(f"🎵 播放 {key} 音轨，来自标签: {label}")
                            last_played[key] = now
        time.sleep(0.05)

except KeyboardInterrupt:
    print("🔚 程序已停止。")
    stream.stop()
