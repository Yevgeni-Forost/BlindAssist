import gtts
import os
from playsound import playsound


def speak(text):
    tts = gtts.gTTS(text, lang="en")
    tts.save("audio.mp3")
    playsound("audio.mp3")
    os.remove("audio.mp3")
