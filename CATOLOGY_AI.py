import joblib
import numpy as np

class CATOLOGY_AI_MODEL:
    def __init__(self,fileNameModel,GUI_INSTANCE):
        self.model = None
        self.modelFileName = fileNameModel
        self.GUI_INSTANCE = GUI_INSTANCE

        self.W1=None
        self.W2=None
        self.W3=None
        self.W4=None
        self.W5=None

        self.b1=None
        self.b2=None
        self.b3=None
        self.b4=None
        self.b5=None


        self.BREED_DICT={}
        self.prepareModel()#load model





    def prepareModel(self):
        self.model = self.load_model_AI()# load from the file and init model.
        self.W1,self.b1=self.model['W1'],self.model['b1']
        self.W2,self.b2=self.model['W2'],self.model['b2']
        self.W3,self.b3=self.model['W3'],self.model['b3']
        self.W4,self.b4=self.model['W4'],self.model['b4']
        self.W5,self.b5=self.model['W5'],self.model['b5']

        self.BREED_DICT={
            0: "Bengal",
            1: "Birman",
            2:"British Shorthair",
            3:"Chartreux",
            4:"European",
            5:"Maine coon",
            6:"Norwegian Forest",
            7:"Beautiful Cat",
            8:"Persian",
        9:"Ragdoll",
        10:"Savannah",
        11:"Siamese",
        12:"Sphynx",
        13:"Turkish Angora",
        14:"Unknown"}


    def load_model_AI(self,filename="model.joblib"):
        try:
            model = joblib.load(filename)
            self.GUI_INSTANCE.display_message_AI(f" Model loaded from {filename}.")
            return model
        except FileNotFoundError as e:
            print(f"File {filename} not found for loading AI model.")
            self.GUI_INSTANCE.stop()
            exit(-1)
        except Exception as e:
            print(e)
            self.GUI_INSTANCE.stop()
            exit(-1)

    def softmax(self,logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward_pass(self,X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):

        a1 = np.dot(X, W1) + b1
        a1 = np.maximum(0, a1)  # ReLU activation

        a2 = np.dot(a1, W2) + b2
        a2 = np.maximum(0, a2)  # ReLU activation

        a3 = np.dot(a2, W3) + b3
        a3 = np.maximum(0, a3)  # ReLU activation

        a4 = np.dot(a3, W4) + b4
        a4 = np.maximum(0, a4)  # ReLU activation

        a5 = np.dot(a4, W5) + b5 # we use softmax later.

        return a1, a2, a3, a4, a5

    def predict(self,X):
        a1, a2, a3, a4, a5 = self.forward_pass(X, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5)
        probabilities = self.softmax(a5)
        predicted_classes = np.argmax(probabilities)
        return predicted_classes

    def WHAT_BREED_IT_IS(self,CHARACTERISTICS):
        BREED=self.predict(CHARACTERISTICS)
        print(f"The chosen breed IS IDX: {BREED}")
        return self.BREED_DICT[BREED]




