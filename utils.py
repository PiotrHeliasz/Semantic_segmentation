
import torch
import torchvision.transforms as transforms



import os
from PIL import Image
from tqdm.auto import tqdm

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



# Set seeds, helper functon to try reproducible things for comparison
def set_seeds(seed: int = 42) :
  '''Sets random sets for torch operations.

  Args:
      seed (int , optional): Random seed to set. Defaults to 42
  '''
  # Set the seed for general torch operations
  torch.manual_seed(seed)
  #Set the seed for CUDA torch operations
  torch.cuda.manual_seed(seed)


def walk_through_dir(dir_path) :
  '''
  Walks through dir_path returning its contents.
  '''
  for dirpath , dirnames , filenames in os.walk(dir_path) :
    print(f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}')
    print('----'*5)



def check_shapes(path):
    '''Checking unique shapes of images in the specified path and counting the occurrences of each shape.

    Args:
        path (str): Path to the directory with images.
    '''
    shape_counts = Counter()  # Using a Counter to store shape occurrences

    for i in tqdm(os.listdir(path)):
        picture = Image.open(f'{path}/{i}')
        picture_array = np.array(picture)
        shape_counts[picture_array.shape] += 1  # Increment the count for this shape

    unique_shapes_list = list(shape_counts.keys())  # Get the list of unique shapes

    print(f'There are {len(unique_shapes_list)} unique shapes in path: {path}')
    print('Unique shapes and their counts:')
    for shape, count in shape_counts.items():
        print(f'Shape: {shape}, Count: {count}')



def check_pixels(path) :
  '''Checking pixels values in random masks (or image) from specified path.
  Args:
    path (str): path to directory with masks (or image).
  Returns:
    Returns number of unique pixel values.
  '''

  unique_values_total = u = 0

  for i in tqdm(os.listdir(path)) :
    mask = Image.open(f'{path}/{i}')
    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)

    if len(unique_values) > u :
      u = len(unique_values)
      unique_values_total = unique_values

  print(f'There is total {u} unique pixel values in path: {path}\nPixel values: {unique_values_total}')
  return u

def check_image_pixels(path) :
  '''Checking particular image pixel values.
  Args:
    path(str): path to particular image
  '''
  unique_values_total = u = 0

  img = Image.open(path)
  img_array = np.array(img)
  unique_values = np.unique(img_array)

  if len(unique_values) > u :
    u = len(unique_values)
    unique_values_total = unique_values

  print(f'There is total {u} unique pixel values\nPixel values: {unique_values_total}')



def count_class_occurrences(mask_dir : str, num_classes: int = 9):
  '''Count class occurrences in given directory.
  Args:
    mask_dir (str): Directory path to check.
    num_classes (int): Number of classes for check.
    By default set to 9.
  '''
  # Initialize dictionaries to count occurrences of each class across all masks:
  class_occurrences = {i: 0 for i in range(num_classes)}

  # Total number of masks counter:
  total_masks = 0

  for mask_filename in tqdm(os.listdir(mask_dir)):
      mask_path = os.path.join(mask_dir, mask_filename)

      mask = np.array(Image.open(mask_path))

      # Check if mask is not empty
      if mask.size == 0:
          continue

      total_masks += 1

      # Check for each class if it exists in the current mask:
      for i in range(num_classes):
          if np.any(mask == i):
              class_occurrences[i] += 1

  # Calculate the percentage of masks containing each class:
  class_percentage = {class_idx : np.round( (count / total_masks) * 100 , 2) for class_idx , count in class_occurrences.items()}

  # Create a pandas DataFrame and histogram:
  df = pd.DataFrame(  list(class_percentage.items()), columns=['Class', 'Percentage']  )

  plt.figure(figsize=(10, 5))
  plt.barh(df['Class'].astype(str), df['Percentage'], color='dodgerblue')

  plt.xlabel('Percentage')
  plt.ylabel('Class' , rotation = 0 , labelpad = 20.)
  plt.title('Percentage of Masks Containing Each Class')
  plt.grid(axis='x', linestyle='--', alpha=0.7 , color = 'black')

  plt.show()

  return class_percentage



def visualize_data(loader , num_images : int = 1):
  '''For visualizing all binary masks for image.
  Args:
  loader: PyTorch Dataloader to load data.
  num_images (int): number of visualizations.
  By dafault num_images = 1.
  '''
  # Load one batch and see example results:
  batch = next(iter(loader))

  image, class_mask = batch[0][0], batch[1][0]

  num_classes = class_mask.size(0)

  for i in range(num_images) :

    image, class_mask = batch[0][i], batch[1][i]

    print("Shape of the image:", image.shape)
    print("Shape of the class mask:", class_mask.shape , '\n')

    # Show unique values in masks:
    for i in range(num_classes):
      print(f"Unique values in class mask {i}: {torch.unique(class_mask[i])}")
    print()

    fig, axes = plt.subplots(1, num_classes + 1, figsize=(18, 6))

    # Show image:
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title("Image")
    axes[0].axis('off')

    # Iterate over masks classes:
    for i in range(num_classes):
        axes[i + 1].imshow(class_mask[i], cmap='gray')
        axes[i + 1].set_title(f"Class {i}")
        axes[i + 1].axis('off')

    plt.show()
    print()

##### SAving and monitoring data ###########


def calculate_class_weights(class_distribution):
    '''Calculate weight for classes base on percentage.
    Args:
        class_distribution (dict): Dictionary with classes and percentages of instances in dataset.
    Returns:
        torch.Tensor: Tensor with weight for every class.
    '''
    total_percentage = sum(class_distribution.values())

    # Reverse the percentages (the fewer examples, the higher the weight)
    class_weights = {cls: total_percentage / percentage for cls, percentage in class_distribution.items()}

    # Normalizujemy wagi, aby suma wynosiła 1
    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

    # Turn into PyTorch tensor:
    class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)

    return class_weights_tensor



def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth'):
    print('Saving checkpoint\n')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),    }, filename)


def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
  if os.path.isfile(filename):
    print('Loading a checkpoint\n')

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']
  else:
    print('Checkpoint file not found. Starting from scratch.\n')

    return 0


def save_model(model: torch.nn.Module,
               target_path: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model (torch.nn.Module): Target PyTorch model to save.
    target_dir (str): Directory for saving the model to.
    model_name (str): Filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="thats_nice_one_model.pth")
  """
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

  # If output_path not exist yet, create one:
  if not os.path.exists(target_path):
    os.makedirs(target_path , exist_ok = True )

  # Create target directory
  model_save_path = os.path.join(target_path , model_name)

  # Save the model state_dict()
  print(f"-> Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)



def create_writer(experiment_name : str ,
                  model_name : str ,
                  extra : str = None ) :

  '''Creates torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specyfic directory'''
  #One experiment one directory

  # Get timestamp of current date in reverse order:
  timestamp = datetime.now().strftime('%Y-%m-%d')

  if extra :
    #Create log directory path:
    log_dir = os.path.join('runs' , timestamp , experiment_name , model_name , extra)
  else :
    log_dir = os.path.join('runs' , timestamp , experiment_name , model_name)

  print(f'Created SummaryWriter saving to {log_dir}\n')
  return SummaryWriter(log_dir = log_dir)




#### EVALUATION ######

def save_loss_curves(results : Dict[str , List[float] ] , plot_path:str = 'loss_curve.png' ) :
  '''Plots training curves of a results dictionary.
  Args:
    results (Dict[str , List[float]): Dictionary of results e.g.
    train/test of loss value and accuracy.
    plot_path (str): Path to save image. By default 'loss_curve.png'
  '''

  # Ensure the directory exists:
  plot_dir = os.path.dirname(plot_path)
  if not os.path.exists(plot_dir) and plot_dir != '':
      os.makedirs(plot_dir)

  # Get the loss values of the results dictionary (training and test)
  loss = results['train_loss']
  test_loss = results['test_loss']

  # Get the accuracy values of the results dictionary (training and test):
  accuracy = results['train_acc']
  test_acc = results['test_acc']

  # Figure out how many epochs there were :
  epochs = range(len(results['train_loss']) )


  # Setup a plot:
  plt.figure(figsize = (15 , 7) )

  # Plotting training and validation loss
  if 'train_loss' in results and 'test_loss' in results:
    plt.subplot(1 ,2 , 1)
    plt.plot(epochs, loss , label = 'train_loss')
    plt.plot(epochs, test_loss , label =  'test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

  # Plotting training and validation accuracy
  if 'train_acc' in results and 'test_acc' in results:
    plt.subplot(1 , 2, 2 )
    plt.plot(epochs , accuracy , label = 'train_accuracy')
    plt.plot(epochs , test_acc , label = 'test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend() ;

  # Save the plot
  plt.savefig(plot_path)
  plt.close()  # Close the figure to free up memory




def rebuild_img(patch_folder: str, base_filename: str, patch_size: int = 256, original_size: int = 1024 , is_it_label: bool = True):
  '''Reconstruct full patchified image. Patches must be named in
  format from preprocessing function which is:

  '{base_filename}_patch_{idx}.tif'

  Where idx is number of patch, from zero to number of patches -1,
  and base_filename is name of originall image.

  Args:
    patch_folder (str): Path to folder with patchified image.
    Path to folder with patches.
    base_filename (str): Name of originall image.
    patch_size(int): Size of every square patch.
    By default patch_size = 256.
    original_size(int): Size of originall image, which will
    be reconstructed. By default original_size = 1024.
    is_it_label(bool): Define the task. To reconstruct mask
    set is_it_label = True. To reconstruct image set
    is_it_label = False. By default is_it_label = True.

  Return:
    Return rebuilded image in form of PyTorch tensor.

  Example Usage:

  rebuild_img(patch_folder = folder_masks_in_patches, base_filename = 'aachen_10' ,
              patch_size = 256 , original_size = 1024 , is_it_label = True)
  '''

  num_patches_per_side = original_size // patch_size
  num_patches = num_patches_per_side * num_patches_per_side

  # Initialize a list, and load patches:
  patches = []
  transform = transforms.Compose([  transforms.PILToTensor()  ])

  for idx in range(num_patches):
      patch_filename = f"{base_filename}_patch_{idx}.tif"
      patch_path = os.path.join(patch_folder, patch_filename)

      # Load the patch and convert to tensor

      patch_image = Image.open(patch_path).convert('L') if is_it_label else Image.open(patch_path)
      patch_tensor = transform(patch_image)  # [CC , H , W ]

      patch_tensor = patch_tensor / 255.  #  to get float values: [0-1]

      # Append the patch tensor to the list
      patches.append(patch_tensor)

  # Stack all patches into a single tensor
  patches = torch.cat(patches, dim=0)  # Shape: (num_patches, patch_size, patch_size)


  # Reshape patches to match the expected shape by fold
  patches = patches.reshape(num_patches , -1).permute(1,0)      # Shape: ( patch_size * patch_size , num_patches)


  # Create fold object and reconstruct the image from patches:
  fold = torch.nn.Fold(output_size=(original_size, original_size), kernel_size=patch_size, stride=patch_size)

  rebuild_img = fold(patches)

  # print(f'Shape of reconstructed image: {rebuild_img.shape}')

  return rebuild_img


### Here write function  to compare predicted binary_masks to ground_truth binary masks >>>
# >>>>
# >>>>
# >>>>
# >>>>


def visualize_predictions(model, dataloader, device, num_images: int =1):
  '''Function to visualize predictions in forms of binary masks.
  Args:
    model: Loaded PyTorch model.
    dataloader: PyTorch dataloader to take example images and masks.
    device: Correct torch.device
    num_images(int): Number of images to visualize, by default = 1.

  '''

  model.eval()
  with torch.no_grad():
      for images, _ in dataloader:
          images = images.to(device)
          outputs = model(images.float())
          predicted_masks = torch.argmax(outputs, dim=1)

          for i in range(num_images):
              image = images[i].cpu().numpy().transpose(1, 2, 0)
              predicted_mask = predicted_masks[i].cpu().numpy()

              # SHow original image:
              plt.figure(figsize=(15, 15))
              plt.subplot(1, 11, 1)
              plt.imshow(image)
              plt.title('Image')
              plt.axis('off')

              # Shwo class masks:
              for class_idx in range(9):
                  class_mask = (predicted_mask == class_idx).astype(float)
                  plt.subplot(1, 11, class_idx + 2)
                  plt.imshow(class_mask, cmap='gray')
                  plt.title(f'Mask {class_idx}')
                  plt.axis('off')

              plt.show()
          break




def show_results(model, loader, device, idx=0):
    """
    Show results.

    Args:
        model (nn.Module): Model UNet.
        loader (DataLoader): Dataloader used for evaluations.
        device (torch.device): correct device.
        idx (int): Index image for visualization, by default idx=0.
    """
    model.eval()
    with torch.no_grad():
        for images, true_masks in loader:
            images = images.to(device)
            true_masks = true_masks.to(device)

            outputs = model(images)
            predicted_masks = torch.argmax(outputs, dim=1)

            # Load correct values:
            image = images[idx].cpu().numpy().transpose(1, 2, 0)
            true_mask = true_masks[idx].cpu().numpy()
            predicted_mask = predicted_masks[idx].cpu().numpy()

            plt.figure(figsize=(15, 5))

            # SHow original image:
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Image')
            plt.axis('off')

            # Ground truth mask:
            true_mask_combined = np.argmax(true_mask, axis=0)  # Argmax of all class masks
            plt.subplot(1, 3, 2)
            plt.imshow(true_mask_combined, cmap='gray')
            plt.title('True Mask')
            plt.axis('off')

            # Predicted mask:
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask,  cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.show()
            break  # just show one image example
