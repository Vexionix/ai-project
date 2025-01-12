import json
import queue

import nltk  # utilities
from langdetect import DetectorFactory, detect
from rake_nltk import Rake
from translate import Translator

##other dependencies
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import wordnet as WORDNET_EN  # we would like to have access to WordNet Facilities

language = ""  # we do not know the language at start-time:)


def DECIDE_LANGUAGE(text):
    try:
        language_dec = detect(text)

        if language_dec == "ro":
            return "ro"
        else:
            return "en"


    except Exception as e:
        print(f"Exception occurred while detecting language. Reason : {e}")
        return "en"


def LOAD_RO_WORDNET():
    with open('wordnet_ro.json', 'r', encoding='utf-8-sig') as wf:
        wordnet_data = json.load(wf)

    return wordnet_data


class Catology_AI:
    def __init__(self, GUI_INSTANCE):

        self.GUI_INSTANCE = GUI_INSTANCE
        self.DECISION_LANGUAGE = "eng"  # default chosen language.

        self.WORDNET_EN = WORDNET_EN
        self.WORNDET_RO = LOAD_RO_WORDNET()

        self.TRANSLATOR_RO_EN = Translator(to_lang="en", from_lang="ro")
        self.TRANSLATOR_EN_RO = Translator(to_lang="ro", from_lang="en")
        self.detector_factory = DetectorFactory()
        self.detector_factory.seed = 0

        self.MESSAGE_QUEUE = queue.Queue()

        self.running = True

    def RECEIVE_TEXT(self, text):
        self.MESSAGE_QUEUE.put(text)

    def GET_TEXT(self):
        text = self.MESSAGE_QUEUE.get()
        return text

    def WRITE_TO_UI(self, text: str):
        self.GUI_INSTANCE.display_message(text)

    def PROCESS_TASKS(self):
        self.GUI_INSTANCE.display_message("Catology_AI has started.")

        while self.running:
            try:
                print("AI_WORKER_AI: Waiting for message...")
                text = self.GET_TEXT() ## this method is blocking the thread. we do not use nowait on the queue.
                ## blocking is good because it saves ressources for the os stand-of-view.
                self.GUI_INSTANCE.display_message(f"AI has got: {text}")

                language_for_this_turn = DECIDE_LANGUAGE(text)
                self.GUI_INSTANCE.display_message(f"LANGUAGE: {language_for_this_turn}")

                DECISION_LANGUAGE= "eng" ## by default
                RAKE_LANGUAGE="english"

                if language_for_this_turn == "ro":
                    DECISION_LANGUAGE="ron"
                    RAKE_LANGUAGE="romanian"

                rake=Rake(language=RAKE_LANGUAGE)

                TOKENS=nltk.word_tokenize(text)
                TOKENS = [token.lower() for token in TOKENS if any(c.isalpha() for c in token)]
                print(f"TOKENS: {TOKENS}")

                self.GUI_INSTANCE.display_message(f"TOKENS: {TOKENS}")






                if text.lower() == "quit":
                    self.running = False
                    break

            except KeyboardInterrupt:
                self.running = False
                break
        self.GUI_INSTANCE.display_message("Catology_AI has ended")
