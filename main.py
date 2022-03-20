import os
import cv2
import AudioOutput as Audio
import SpeechRecognition as Sr
import ClassifyRoom as Room
import ObjectDetection as Object
import BFMatcher as Bf
import Camera
import ViolaJones as Vj

voice = 'null'


def there_exists(terms, voice):
    for term in terms:
        if term in voice:
            return True


def respond(voice_data):
    ready = 'ready'

    if there_exists(['hey', 'hello'], voice_data):
        Audio.speak('Hello, what can I help you with')

    # We ended up not using classify_object. This part was dynamically connected to the Object classification module,
    # and by saying the name of the object you look for, the relevant model was loaded and the search was performed.
    #
    # if there_exists(["where is my", "find my", "looking for my"], voice_data):
    #     index = -1
    #     if there_exists('hat', voice_data):
    #         index = 0
    #     elif there_exists('key', voice_data):
    #         index = 1
    #     elif there_exists('wallet', voice_data):
    #         index = 2
    #     else:
    #          Audio.Speak("That is not an item I can search")
    #     if index != -1:
    #         classify_object(index)

    if there_exists(['where am I', 'room', 'location'], voice_data):
        Audio.speak("ok prepare your camera, tell me when you are ready")
        answer = Sr.Listen()
        if ready in answer:
            room = Room.classify_rooms()
            output = 'you are in the ' + room
            Audio.speak(output)
        return

    if there_exists(['add', 'item'], voice_data):
        cap = cv2.VideoCapture(Camera.address)
        Audio.speak("ok prepare your camera, tell me when you are ready")
        answer = Sr.Listen()
        if ready in answer:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    Audio.speak('What would you like to call it?')
                    item = Sr.Listen()
                    cv2.imwrite(os.path.join('BfImages', item + '.jpg'), frame)
                cap.release()
                cv2.waitKey(2000)
            Audio.speak('All done')
            cap.release()
        return

    if there_exists(['face'], voice_data):
        Audio.speak("ok prepare your camera, tell me when you are ready")
        answer = Sr.Listen()
        if ready in answer:
            result = Vj.face_detection()
            if result == 'null':
                Audio.speak("I can't detect more than one face, or there is no face in the photo")
            else:
                Audio.speak('All done')
        return

    if there_exists(['who', 'in front'], voice_data):
        Audio.speak("ok prepare your camera, tell me when you are ready")
        answer = Sr.Listen()
        names = []
        if ready in answer:
            names = Vj.detect_faces()
        if not names:
            Audio.speak("I do not recognize anybody in the picture.")
        for name in names:
            Audio.speak(name + "is in front of you")
        return

    if there_exists(['find'], voice_data):
        objects = ['dog', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'sports ball', 'bottle',
                   'remote', 'laptop', 'cell phone', 'book', 'toothbrush']
        rel_object = ""
        for obj in objects:
            if obj in voice_data:
                rel_object = obj
        if rel_object != "":
            Audio.speak("ok prepare your camera, tell me when you are ready")
            answer = Sr.Listen()
            cap = cv2.VideoCapture(Camera.address)
            if ready in answer:
                if cap.isOpened():
                    ret, frame = cap.read()
                    cv2.waitKey(2000)
                    if Object.detect_object(rel_object, frame):
                        output = 'I can see the ' + rel_object + ' in the picture'
                        Audio.speak(output)
                    else:
                        Audio.speak("I can't find your " + rel_object)
            else:
                Audio.speak("I can't search for that object")
            cap.release()
        return

    if there_exists(['what is that'], voice_data):
        Audio.speak("ok prepare your camera, tell me when you are ready")
        answer = Sr.Listen()
        if ready in answer:
            result = Bf.bf_matcher()
            output = 'You are holding ' + result
            Audio.speak(output)
        return

    if there_exists(["exit", "quit", "bye"], voice_data):
        Audio.speak("Good bye")
        exit()

    Audio.speak("I didnt understand")
    Audio.speak("Please try again.")


def main():
    if not os.path.isdir('Test'):
        os.mkdir('Test')
    if not os.path.isdir('known_faces'):
        os.mkdir('known_faces')
    Audio.speak('Hello, welcome to blind assist')
    while True:
        Audio.speak('i am listening')
        voice = Sr.Listen()
        while voice == 'null':
            Audio.speak('Sorry, I did not get that')
            Audio.speak('Please, try speaking again')
            Audio.speak('listening')
            voice = Sr.Listen()
        respond(voice)


if __name__ == "__main__":
    main()
