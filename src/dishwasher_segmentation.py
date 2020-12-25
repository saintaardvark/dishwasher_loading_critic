#!/usr/bin/env python

# Following along with: https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
# Generated by https://traingenerator.jrieke.com/

# Before running, install required packages:
# pip install numpy torch torchvision pytorch-ignite

from pathlib import Path
import shutil
import urllib
import zipfile

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data/external/image-data"
# ----------------------------------- Setup -----------------------------------
# INSERT YOUR DATA HERE
# Expected format: One folder per class, e.g.
# train
# --- dogs
# |   +-- lassie.jpg
# |   +-- komissar-rex.png
# --- cats
# |   +-- garfield.png
# |   +-- smelly-cat.png
#
# Example: https://github.com/jrieke/traingenerator/tree/main/data/image-data
example_url = "https://github.com/jrieke/traingenerator/raw/main/data/fake-image-data.zip"
train_data = DATA_DIR # required
val_data = DATA_DIR   # optional
test_data = None      # optional

use_cuda = torch.cuda.is_available()

def download_data():
    """Download data if needed
    """
    # COMMENT THIS OUT IF YOU USE YOUR OWN DATA.
    # Download example data into ./data/image-data (4 image files, 2 for "dog", 2 for "cat").
    dpath = Path(DATA_DIR)
    if not (dpath.exists()):
        zip_path, _ = urllib.request.urlretrieve(example_url)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(DATA_DIR)
    # Manual cleanup
    osx_junkdir = (dpath / "__MACOSX")
    if osx_junkdir.exists():
        shutil.rmtree(osx_junkdir)

def preprocess(data, name, batch_size):
    """Preprocess data
    """
    if data is None:  # val/test can be empty
        return None

    # Read image files to pytorch dataset.
    transform = transforms.Compose([

        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data, transform=transform)

    # Wrap in data loader.
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(name=="train"), **kwargs)
    return loader

def log_results(name, metrics, epoch):
    print(
        f"{name + ':':6} loss: {metrics['loss']:.3f}, "
        f"accuracy: {metrics['accuracy']:.3f}"
    )

def main():
    """Main entry point
    """
    download_data()

    # Set up hyperparameters.
    lr = 0.001
    batch_size = 128
    num_epochs = 3

    # Set up logging.
    print_every = 1  # batches

    # Set up GPU.
    device = torch.device("cuda" if use_cuda else "cpu")

    # ------------------------------- Preprocessing -------------------------------
    train_loader = preprocess(train_data, "train", batch_size)
    val_loader = preprocess(val_data, "val", batch_size)
    test_loader = preprocess(test_data, "test", batch_size)

    # ----------------------------------- Model -----------------------------------
    # Set up model, loss, optimizer.
    # Note: putting model into eval mode right away.
    model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    # model = model.to(device)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # Our test image
    cat = '/home/aardvark/best_cat.jpg'
    dog = '/home/aardvark/dog.jpg'
    dishwasher = '/home/aardvark/dishwasher.jpg'
    dog_park = '../data/pexels-photo-1485799.jpeg'
    man = '../data/pexels-photo-5648380.jpeg'
    family_in_masks = '../data/pexels-photo-4127449.jpeg'
    woman_in_supermarket = '../data/pexels-photo-4177708.jpeg'
    img = Image.open(woman_in_supermarket)

    plt.imshow(img)
    plt.show()

    trf = transforms.Compose([transforms.Resize(256),
                              # transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                   std = [0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0)

    # Now to do a forward pass
    out = model(inp)['out']
    print(out.shape)

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(om.shape)
    print(np.unique(om))

    # Define helper function
    def decode_segmap(image, nc=21):
        """Decode segmentation map as appropriate
        """
        label_colors = np.array([(0, 0, 0), # 0=background,
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
                                 ])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    rgb = decode_segmap(om)
    plt.imshow(rgb)
    plt.show()
    # # --------------------------------- Training ----------------------------------
    # # Set up pytorch-ignite trainer and evaluator.
    # trainer = create_supervised_trainer(
    #     model,
    #     optimizer,
    #     loss_func,
    #     device=device,
    # )
    # metrics = {
    #     "accuracy": Accuracy(),
    #     "loss": Loss(loss_func),
    # }
    # evaluator = create_supervised_evaluator(
    #     model, metrics=metrics, device=device
    # )

    # @trainer.on(Events.ITERATION_COMPLETED(every=print_every))
    # def log_batch(trainer):
    #     batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    #     print(
    #         f"Epoch {trainer.state.epoch} / {num_epochs}, "
    #         f"batch {batch} / {trainer.state.epoch_length}: "
    #         f"loss: {trainer.state.output:.3f}"
    #     )

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_epoch(trainer):
    #     print(f"Epoch {trainer.state.epoch} / {num_epochs} average results: ")

    #     # Train data.
    #     evaluator.run(train_loader)
    #     log_results("train", evaluator.state.metrics, trainer.state.epoch)

    #     # Val data.
    #     if val_loader:
    #         evaluator.run(val_loader)
    #         log_results("val", evaluator.state.metrics, trainer.state.epoch)

    #     # Test data.
    #     if test_loader:
    #         evaluator.run(test_loader)
    #         log_results("test", evaluator.state.metrics, trainer.state.epoch)

    #     print()
    #     print("-" * 80)
    #     print()

    # # Start training.
    # trainer.run(train_loader, max_epochs=num_epochs)

if __name__ == '__main__':
    main()
