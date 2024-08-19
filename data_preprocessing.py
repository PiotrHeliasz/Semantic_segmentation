
import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn

from tqdm.auto import tqdm
import os
from PIL import Image



def preprocess_data(path: str , output_path: str, check_image: bool = False ) :
  '''Preprocess images/masks of size: [1000,1000] by zero padding up to [1024,1024],
  and transforms into patches of size [256,256]. Discard rest of images.

  Args:
    path (str): Path to the directory with whole images/masks.
    output_path (str): Path to the directory where processed patches will be saved.
    check_image (bool): Make just 16 patches and break if True. By default check_image = False.

  Return:
    output_path is returned.
  '''

  # If output_path not exist yet, create one:
  if not os.path.exists(output_path):
    os.makedirs(output_path)


  # loop through file with data:
  for filename in tqdm(os.listdir(path) ) :
    image_path = os.path.join(path , filename )


    #Open image and turn it into torch tensor:
    img = Image.open(image_path)
    tensor = transforms.PILToTensor()(img)    # [CC, H , W]


    if tensor.shape in [(1,1000,1000) , (3,1000,1000) ] :
      target_height = target_width = 1024

      pad_height = target_height - tensor.shape[1]
      pad_width = target_width - tensor.shape[2]

      pad_top_bottom = pad_height // 2
      pad_right_left = pad_width // 2

      tensor = nn.functional.pad(tensor,
                                (pad_right_left, pad_right_left, pad_top_bottom, pad_top_bottom),
                                mode='constant', value=0)



    if tensor.shape in [(1,1024,1024) , (3,1024,1024) ] :   # Discard tesnors different than 1024 , 1000

      patches = tensor.unfold(1, 256, 256).unfold(2, 256, 256)
      patches = patches.permute(1,2,0,3,4,)  # num_patches_h(4), num_patches_w(4),  , CC , 256, 256

      # Determine the number of channels
      num_channels = patches.shape[2]  # This will be 1 for grayscale and 3 for RGB

      patches = patches.reshape(-1 , num_channels , 256, 256)


      base_filename = os.path.splitext(filename)[0]
      for i, patch in enumerate(patches):

        patch_filename = f"{base_filename}_patch_{i}.tif"
        patch_path = os.path.join(output_path, patch_filename)

        # print(f'This is base_filename: {base_filename},\nThis is patch_filename: {patch_filename},\nThis is patch_path: {patch_path}')

        save_image(patch /255.  , patch_path)

      if check_image :
        break

  return output_path

