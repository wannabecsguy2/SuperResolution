import torch
import numpy as np
import glob as glob

from torch.utils.data import DataLoader, Dataset
from PIL import Image

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1

class SRCNNDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.all_image_paths = glob.glob(f'{image_paths}/*')
        self.all_label_paths = glob.glob(f'{label_paths}/*')

    def __len__(self):
        return (len(self.all_image_paths))
    
    def __getitem__(self, index):
        image = Image.open(self.all_image_paths[index]).convert('RGB')
        label = Image.open(self.all_labels_paths[index]).convert('RGB')

        image = np.array(image, dtype = np.float32)
        label = np.array(label, dtype = np.float32)

        image /= 255
        label /= 255

        image = image.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])

        return(
            torch.tensor(image, dtype = torch.float),
            torch.tensor(label, dtype = torch.float)
        )

def get_datasets(
    train_image_paths, train_label_paths, 
    val_image_paths, val_label_paths
):
    dataset_train = SRCNNDataset(train_image_paths, train_label_paths)
    dataset_val = SRCNNDataset(val_image_paths, val_label_paths)

    return dataset_train, dataset_val

def get_dataloaders(dataset_train, dataset_val):
    train_loader = DataLoader(
        dataset_train,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False
    )

    return train_loader, val_loader
