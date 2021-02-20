import os
from pathlib import Path

if 'COLAB_GPU' in os.environ:
    print("Running on Colab!")
    COLAB = True
    from google.colab import drive
    drive.mount('/content/gdrive')
    dataset_root = Path('/content/gdrive/MyDrive/dishwasher_training_data')
else:
    print("Nope, not on Colab")
    COLAB = False
    dataset_root = Path('/home/aardvark/dev/dishwasher_training_data/roboflow')
