import numpy as np
import pickle
import json


class Main(object):
    def __init__(self):
        # loads model
        self.model = None
        with open('./model/linear_model.pickle', 'rb') as model_in:
            self.model = pickle.load(model_in)

    def predict(self, input_arg):
        # say input_arg is a number but the type it contains is string
        input_arg = float(json.loads(input_arg))
        x_test = np.array([[input_arg]])
        result = self.model.predict(x_test)
        return json.dumps(str(result[0,0]))