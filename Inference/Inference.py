import torch


class Inference:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        # Perform inference using the model
        with torch.no_grad():    
            return self.model(input_data)
      
    