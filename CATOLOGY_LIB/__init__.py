import json
import os
import sys
import threading
import time
import tkinter as tk
from io import StringIO

import random
from translate import Translator # for translation


def LOAD_RO_WORDNET():
    with open('wordnet_ro.json', 'r',encoding='utf-8-sig') as wf:
        wordnet_data=json.load(wf)

    return wordnet_data