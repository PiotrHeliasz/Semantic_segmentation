'''
Constains functionallity for creating PyTorch DataLoader's for
image classification data.
'''
import os
from PIL import Image

from torchvision import datasets , transforms
from torch.utils.data import DataLoader , Dataset
import torchvision.transforms as transforms
import torch


BATCH_SIZE = 32
# NUM_CLASSES = check_pixels(label_path_train)     #---> For check
NUM_CLASSES = 9
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True

transform = transforms.Compose([
    transforms.PILToTensor()
])


def create_dataloaders(train_labels_path: str ,
                       train_images_path: str,
                       test_labels_path: str ,
                       test_images_path: str ,
                       transform : transforms.Compose ,
                       batch_size: int= BATCH_SIZE ,
                       num_workers : int = NUM_WORKERS ,
                       num_classes: int = NUM_CLASSES,
                       pin_memory: bool = PIN_MEMORY ) :
  '''Creates training and testing Datasets and DataLoaders.

  Takes in a training directory and testing directory path and turns them into
  PyTorch datasets and then into PyTorch DataLoaders

  Args:
    train_dir: PAth to training directory.
    test_dir: Path to testing direcotry.
    transform: torchvision transforms to perfom on training and testing data.
    batch_size: Number of samples per batch in eahch of DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
    pin_memory: If True, pin memory for faster computation.

  Returns:
    A tuple of (train_dataloader , test_dataloader , train_dataset , test_dataset ).
  '''


  class CustomDataset(Dataset) :
    def __init__(self, image_dir , mask_dir , num_classes ,transform : None ) :
      self.image_dir = image_dir
      self.mask_dir = mask_dir
      self.transform = transform

      self.image_paths = [os.path.join(image_dir , fname) for fname in os.listdir(image_dir) ]
      self.mask_paths = [os.path.join(mask_dir , fname) for fname in os.listdir(mask_dir) ]

      self.num_classes = num_classes

    def __len__(self) :
      return len(self.image_paths) # smae number of images and masks

    def __getitem__(self, idx) :
      #Load image:
      image_path = self.image_paths[idx]
      image = Image.open(image_path).convert('RGB')

      #Load mask:
      mask_path = self.mask_paths[idx]
      mask = Image.open(mask_path).convert('L')

      #Apply transformation if specified:
      if self.transform :
        image = self.transform(image )
        mask = self.transform(mask )

        # mask = mask.to(torch.float32)        # Mask are float default
        image = image.to(torch.float32)/255.

      mask = torch.squeeze(mask)    # SQUEEZE additional 1

      # Change mask into correct multiclass masks format:
      class_mask = torch.zeros((self.num_classes, mask.shape[0] , mask.shape[1] ) )
      # print(mask.shape)

      for i in range(self.num_classes) :
        class_mask[i] = (mask == i).float()     # SAME AS CLASS_MASK[i, : , : ] = ......

      return image, class_mask


  #Create datasets and dataloaders:
  train_dataset = CustomDataset(image_dir=train_images_path, mask_dir=train_labels_path, num_classes= NUM_CLASSES, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True , num_workers= NUM_WORKERS , pin_memory= PIN_MEMORY)

  val_dataset = CustomDataset(image_dir=test_images_path, mask_dir=test_labels_path, num_classes = NUM_CLASSES, transform=transform)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False , num_workers= NUM_WORKERS , pin_memory= PIN_MEMORY)

  return train_loader , val_loader , train_dataset , val_dataset
