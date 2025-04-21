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
BLOCK_DURATION = 0.25
CHANNELS = 1
THRESHOLD = 0.2
COOLDOWN = 0.5
BUFFER = queue.Queue()
last_played = {}
active_loops = {}
DEACTIVATION_TIMEOUT = 2.0

# ==== 音频播放初始化（使用多个通道支持同时播放） ====
pygame.mixer.init()
pygame.mixer.set_num_channels(16)  # 设置可同时播放的最大通道数

sound_map_once = {
    "Car horn": "sounds/car_drums.wav",
    "Beep": "sounds/beep_click.wav"
}
sound_map_loop = {
    "Speech": "Forest_sounds/bird.wav",
    "Traffic noise": "sounds/traffic_rhythm.wav"
}

loaded_once = {label: pygame.mixer.Sound(path) for label, path in sound_map_once.items()}
loaded_loop = {label: pygame.mixer.Sound(path) for label, path in sound_map_loop.items()}
loop_channels = {}  # 存储正在播放的持续声音的通道

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

print("🎧 正在监听城市声音... 按 Ctrl+C 停止。")

try:
    while True:
        if not BUFFER.empty():
            audio_block = BUFFER.get()
            waveform = np.reshape(audio_block, (-1,)).astype(np.float32)

            scores, embeddings, spectrogram = yamnet_model(waveform)
            mean_scores = np.mean(scores.numpy(), axis=0)
            top_indices = np.argwhere(mean_scores > THRESHOLD).flatten()

            now = time.time()
            detected_labels = set()

            for idx in top_indices:
                label = class_names[idx]

                for key in sound_map_once:
                    if key.lower() in label.lower():
                        if (key not in last_played) or (now - last_played[key] > COOLDOWN):
                            loaded_once[key].play()
                            print(f"🔔 播放一次性音轨: {key}（标签: {label}）")
                            last_played[key] = now

                for key in sound_map_loop:
                    if key.lower() in label.lower():
                        detected_labels.add(key)
                        if key not in active_loops:
                            channel = pygame.mixer.find_channel()
                            if channel:
                                channel.play(loaded_loop[key], loops=-1)
                                loop_channels[key] = channel
                                print(f"🎵 开始持续播放: {key}（标签: {label}）")
                            active_loops[key] = now
                        else:
                            active_loops[key] = now

            expired = [key for key, t in active_loops.items() if now - t > DEACTIVATION_TIMEOUT]
            for key in expired:
                if key in loop_channels:
                    loop_channels[key].stop()
                    print(f"⏹️ 停止持续播放: {key}")
                    del loop_channels[key]
                del active_loops[key]

        time.sleep(0.05)

except KeyboardInterrupt:
    print("🔚 程序已停止。")
    stream.stop()
    for channel in loop_channels.values():
        channel.stop()
