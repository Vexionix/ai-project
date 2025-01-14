import json
import os
import queue

import joblib
import nltk  # utilities
import numpy as np
from openai import OpenAI

from langdetect import DetectorFactory, detect
from rake_nltk import Rake
from translate import Translator

from CATOLOGY_AI import CATOLOGY_AI_MODEL

##other dependencies
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')
from openai import OpenAI

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
        self.MODEL_AI =None

        self.X = None  ## features of our cat
        self.description_data=None

        self.WORD_DICT = {
            "sex": {
                "female": 0,
                "male": 1
            },
            "age": {
                "between 1 and 2 years": 0,
                "between 2 and 10 years": 1,
                "less than 1 month": 2,
                "more than 10 years": 3
            },
            "number": {
                "has been found with no cats friends": 0,
                "has been found with one cat friend": 1,
                "has been found with two cats friends": 2,
                "has been found with three cats friends": 3,
                "has been found with 4 kitties friends": 4,
                "has been found with more than 5 cats friends": 5
            },
            "house_type": {
                "an apartment without balcony": 0,
                "an apartment with balcony or terrace": 1,
                "an individual house": 2,
                "a house in a subdivision": 3
            },
            "zone": {
                "semi-urban zone": 0,
                "urban zone": 1,
                "rural zone": 2
            },
            "outdoor": {
                "does not go outside": 0,
                "rarely goes outside for an hour": 1,
                "goes moderately outside, 1-5 hours per day": 2,
                "goes outside for more than 5 hours per day": 3,
                "it's outside all the time, coming back just to eat": 4
            },
            "obs": {
                "this cat does not need observation often": 1,
                "this cat needs attention for at least one hour": 2,
                "this cat needs attention between one and 5 hours per day": 3,
                "this cat needs all your time": 4
            },
            "abondance": {
                "around aren't any trees, grass, or bushes": 1,
                "there is a small presence of nature around": 2,
                "there's full of green life around": 3,
                "it's not clearly its favorite environment": 4
            },
            "catches_birds": {
                "never": 0,
                "rarely (1 to 5 times a year)": 1,
                "sometimes (5 to 10 times a year)": 2,
                "often (1 to 3 times a month)": 3,
                "very often (once a week or more)": 4
            },
            "catches_mammals": {
                "never": 0,
                "rarely (1 to 5 times a year)": 1,
                "sometimes (5 to 10 times a year)": 2,
                "often (1 to 3 times a month)": 3,
                "very often (once a week or more)": 4
            },

        }

        self.ATTRIBUTES_TO_SET = 8
        self.ATTRIBUTES_SET = 0

        # invert the original WORD_DICT to map numbers to propositions
        self.INVERTED_WORD_DICT = {
            "sex": {v: k for k, v in self.WORD_DICT["sex"].items()},
            "age": {v: k for k, v in self.WORD_DICT["age"].items()},
            "number": {v: k for k, v in self.WORD_DICT["number"].items()},
            "house_type": {v: k for k, v in self.WORD_DICT["house_type"].items()},
            "zone": {v: k for k, v in self.WORD_DICT["zone"].items()},
            "outdoor": {v: k for k, v in self.WORD_DICT["outdoor"].items()},
            "obs": {v: k for k, v in self.WORD_DICT["obs"].items()},
            "abondance": {v: k for k, v in self.WORD_DICT["abondance"].items()},
            "catches_birds": {v: k for k, v in self.WORD_DICT["catches_birds"].items()},
            "catches_mammals": {v: k for k, v in self.WORD_DICT["catches_mammals"].items()},
        }

        with open("descriptions.json") as fd:
            self.description_data=fd.read()

        self.CATS_DESCRIPTIONS=None

        self.CATS_DESCRIPTIONS=json.loads(self.description_data)

        self.CATS_DESCRIPTIONS = {item['breed']: item['description'] for item in self.CATS_DESCRIPTIONS['cat_breeds']}

    def generate(self, cat_name):
        if not self.MODEL_AI.is_cat(cat_name):
            self.WRITE_TO_UI("Cat doesn't exist")
        else:
            print(self.CATS_DESCRIPTIONS)
            self.WRITE_TO_UI(f"Generated description: {self.CATS_DESCRIPTIONS[cat_name.lower()]}")


    def load_model_AI(self,filename="trained_model.joblib"):

        model = joblib.load(filename)
        print(f"WORKER: Model loaded from {filename}")
        return model

    def RECEIVE_TEXT(self, text):
        self.MESSAGE_QUEUE.put(text)

    def GET_TEXT(self):
        text = self.MESSAGE_QUEUE.get()
        return text

    def WRITE_TO_UI(self, text: str):
        self.GUI_INSTANCE.display_message_AI(text)

    def generate_generic_description(self, cat_name, X):
        sex_value = X[0][0]
        print(f"Sex value: {sex_value}")
        return f"The cat {cat_name} is a {self.INVERTED_WORD_DICT['sex'][sex_value]}"

    def set_features(self, tokens, language_got, X):
        for token in tokens:
            # Get synsets for the token in the specified language
            synsets = WORDNET_EN.synsets(token, lang=language_got)

            # Check hypernyms for each synset
            for synset in synsets:
                for hypernym in synset.hypernyms():

                    print(f"Token: {token}, Hypernym: {hypernym.name()}")


                    for category, mapping in self.WORD_DICT.items():
                        for key, value in mapping.items():
                            if key.lower() in hypernym.name().lower():  # Case-insensitive match
                                # Update the corresponding feature in X (assuming X is a 2D array with features)
                                feature_index = self.get_feature_index(category)
                                if feature_index is not None:
                                    X[0][feature_index] = value  # Set the appropriate feature

    def get_feature_index(self, category):
        # Assuming we have a fixed feature index for each category in the WORD_DICT
        print("Getting feature index for category:", category)

        feature_indices = {
            "sex": 0,
            "age": 1,
            # Add indices for other features here
        }
        return feature_indices.get(category, None)

    def extract_traits_from_keywords(self, keywords):
        try:
            client = OpenAI(###)
            system_message = "You are a mapping assistant that maps keywords to predefined categories and values."

            user_message = f"""
                    Here is the dictionary defining the categories and their values:
                    {self.WORD_DICT}

                    These are the keywords describing the cat's traits: {keywords}.
                    Map each keyword to the most appropriate category and value from the dictionary.
                    Return the result in a JSON format, where each category is a key and the matched value is its value.
                    If a keyword does not fit any category, exclude it from the result.
                """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )

            result = response.choices[0].message.content.strip()

            return json.loads(result)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}

    def parse_traits(self, traits_text):
        if isinstance(traits_text, dict):
            return traits_text

        try:
            traits_dict = json.loads(traits_text)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
            return {}

        return traits_dict

    def PROCESS_TASKS(self):

        self.MODEL_AI=CATOLOGY_AI_MODEL("model.joblib",self.GUI_INSTANCE) ## pass.

        self.WRITE_TO_UI("Catology_AI has started.")
        X = np.zeros(26)  # Shape (26,)

        # If you need it to be in a 2D array with 1 sample (1, 26)
        X = X.reshape(1, 26)




        breeds=self.MODEL_AI.WHAT_BREED_IT_IS(X)
        self.WRITE_TO_UI(f"Breed is {breeds}")
        self.WRITE_TO_UI(self.generate_generic_description('Bengal', X))




        iteration=0
        while self.running:
            try:
                print("AI_WORKER_AI: Waiting for message...")
                text = self.GET_TEXT()## this method is blocking the thread. we do not use nowait on the queue.
                ## blocking is good because it saves ressources for the os stand-of-view.
                self.WRITE_TO_UI(f"AI has got: {text}")
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

                self.WRITE_TO_UI(f"TOKENS: {TOKENS}")
                rake_analysys=rake.extract_keywords_from_text(text)

                rake_keywords=rake.get_ranked_phrases()
                print(f"MAIN_TOKENS: {rake_keywords}")




                self.set_features(rake_keywords,DECISION_LANGUAGE,X)
                print(f"Features vect:{X}")


                if TOKENS.__len__()<6:
                    self.WRITE_TO_UI("Can you please tell more about your cat in your sentence?")
                    iteration=iteration+1
                else:
                    traits_mapping = self.extract_traits_from_keywords(TOKENS)
                    traits_dict = self.parse_traits(traits_mapping)
                    print(traits_dict)
                    for category, trait_value in traits_dict.items():
                        if category in self.WORD_DICT:
                            options = self.WORD_DICT[category]
                            if trait_value in options.values():
                                category_index = list(self.WORD_DICT.keys()).index(category)
                                X[0][category_index] = trait_value
                    print(X)
                    breeds = self.MODEL_AI.WHAT_BREED_IT_IS(X)
                    self.WRITE_TO_UI(f"Breed is {breeds}")
                    X = np.zeros(26)
                    X = X.reshape(1, 26)
                    iteration=0





                if text.lower() == "quit":
                    self.running = False
                    break

            except KeyboardInterrupt:
                self.running = False
                break
        self.WRITE_TO_UI("Catology_AI has ended")


