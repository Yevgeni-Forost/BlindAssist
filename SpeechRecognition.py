import speech_recognition as sr
import AudioOutput as Audio
recognize = sr.Recognizer()


def Listen():
    with sr.Microphone() as source:
        Audio.speak("Speak Now")
        audio = recognize.listen(source)

        try:
            # print("HI")
            text = recognize.recognize_google(audio)
            if text != 'null:':
                print(text)
            else:
                print("text is null")
            return text
        except:
            return 'null'
