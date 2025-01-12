import nltk
from rake_nltk import Rake

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import wordnet
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
import random
from translate import Translator

DetectorFactory.seed = 0  # consistent results:)
language = ""  # we do not know the language at start-time:)


def generate_sentence_from_keyword(keyword, lang='eng'):
    synsets = wordnet.synsets(keyword, lang=lang)  ## we search thru the synsets..
    if not synsets:
        if lang == 'ron':
            return f"N-am gasit niciun sens pentru cuvantul {keyword}."
        else:
            return f"Could not find any sense for {keyword}."

    synset = synsets[0]  # we set with 1st at start

    # we look in all synsets and try to find one which has a proposition for us:)

    for syn in synsets:
        if syn.examples(lang=lang):
            synset = syn
            break  # we stop we do not care more:)

    definition = synset.definition(lang=lang)  # definition of the word from wordnet:)
    example_sentences = synset.examples(lang=lang)  # a description from the synset ->our sentence

    # using the first example sentence if available, otherwise creating our own
    if example_sentences:
        sentence = example_sentences[0]
    else:
        if lang == 'ron':
            sentence = f"{keyword.capitalize()} este definit ca {definition}."  # we create one basic if it is not available something else
        else:
            sentence = f"{keyword.capitalize()} is defined {definition}"

    return sentence


def get_synonyms_antonyms(word, lang='eng'):
    synonyms = set()
    antonyms = set()
    hypernyms = set()
    negation_word = "no "

    if lang == 'ron':
        negation_word = "nu "

    for syn in wordnet.synsets(word, lang=lang):
        for lemma in syn.lemmas(lang=lang):
            synonyms.add(lemma.name())
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    antonyms.add(negation_word + ant.name())
        for hyper in syn.hypernyms():
            hypernyms.update(hyper.lemma_names(lang=lang))
    return list(synonyms), list(antonyms), list(hypernyms)


def replace_words(words, ratio=0.2, lang='eng'):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # we put the filtered tokens we do not care to replace punctuation..
    print(f"We got {len(words)} \n "
          f"words: {words} \n")
    number_of_words_to_replace = int(len(words) * ratio)
    print(f"Number of words to replace: {number_of_words_to_replace}")
    indices_to_replace = random.sample(range(len(words)), number_of_words_to_replace)
    print(f"Random indices to replace: {indices_to_replace}")
    new_words = words[:]  ##this is how we copy exactly

    for i in indices_to_replace:
        word = words[i]
        synonyms, antonyms, hypernyms = get_synonyms_antonyms(word, lang=lang)
        print(f"For the word : {word} \n"
              f"The synonyms : {synonyms} \n"
              f"The antonyms : {antonyms} \n"
              f"The hypernyms : {hypernyms} \n")
        replacement = synonyms + antonyms + hypernyms
        if replacement:
            new_words[i] = random.choice(replacement)
            print(f"For the word : {word} \n"
                  f"The replacement is : {new_words[i]} \n")

    return " ".join(new_words)


if __name__ == "__main__":

    print("Homework 3: Knowledge Recognition")
    running = True
    print("Please say something...\nInput 'quit' when you are done.")


    iteration = 0

    while running:
        result = input(">>:")
        if result == "":
            print("Nothing has been introduced!. Retry!!")
            continue

        if result.lower() == "quit":
            running = False
            break

        try:
            language = detect(result)
            print(f"Detected language: {language}")
        except Exception as e:
            print(f"Could not detect language.Enter more text please.. \n"
                  f"Reason : {e} \n")

        tokens = nltk.word_tokenize(result)  # Tokenize the input->this also remove stopwords:)

        filtered_tokens = [token for token in tokens if any(c.isalpha() for c in token)]  # this remove numbers:)
        print(f"Filtered tokens: {filtered_tokens}")

        # we calculate the frequency of how many of length x words have we got.
        token_length = [len(token) for token in filtered_tokens]


        ##stylometric analysis here
        NUM_WORDS = len(filtered_tokens)  ##how many relevant words we got.
        NUM_CHARS = sum(len(token) for token in filtered_tokens)  ## how many characters

        ##frequency analysis..
        WORD_FREQ = {}

        for token in filtered_tokens:
            if token in WORD_FREQ:
                WORD_FREQ[token] += 1
            else:
                WORD_FREQ[token] = 1

        print(f"Number of words: {NUM_WORDS}")
        print(f"Number of characters: {NUM_CHARS}")
        print(f"Word frequencies: {WORD_FREQ}")



        print(f"Original text: {result}")
        print("Generating alternative text:")
        ## we need to use to_lower on tokens for better results...
        filtered_tokens_lowered = [token.lower() for token in filtered_tokens]

        decision_language = "eng"
        rake_language = "english"
        translator = Translator(to_lang="en", from_lang="ro")  ## a backup plan for better accuracy
        translator_en_ro = Translator(to_lang="ro", from_lang="en")
        ## we only want english and romanian
        if language == "ro":
            decision_language = "ron"
            rake_language = "romanian"

        rake = Rake(language=rake_language)

        print(f"The decision language is : {decision_language}")
        list_of_alternative_texts = []

        for _ in range(3):
            alternative_text = replace_words(filtered_tokens_lowered, ratio=1.0, lang=decision_language)
            list_of_alternative_texts.append(alternative_text)

        print(f"Alternative texts: ")

        number = 1
        for alternative_text in list_of_alternative_texts:
            print(f"Alternative text {number}: {alternative_text}")
            number += 1

        rake.extract_keywords_from_text(result)
        keywords_got = rake.get_ranked_phrases()
        print(f"Keywords got: {keywords_got}")

        ## this is how we generate:)

        for k in keywords_got:
            list_of_keywords = k.split()  # we separate each word because there may be composed.
            print(f"Keywords: {list_of_keywords}")

            # as wordnet is more on english we try to convert to en->generate->convert to ro.
            if decision_language == "ron":
                for k_from_bag in list_of_keywords:
                    translation = translator.translate(k_from_bag)  ## ro-en
                    print(f"Translation of {k_from_bag}: {translation}")
                    generated_from_translation = generate_sentence_from_keyword(translation, 'eng')
                    print(f"Alternative text generated: {generated_from_translation}")
                    final_text_generated = translator_en_ro.translate(generated_from_translation)  # back to en-ro
                    print(f"Final translation:{final_text_generated}")

                # standard
            for k_from_bag in list_of_keywords:  # we take each token individually
                print(f"--->For keyword {k_from_bag}: {generate_sentence_from_keyword(k_from_bag, decision_language)}")

        iteration += 1  # we keep track of how many iterations has the user has provided us text

    print("Done running!!")
