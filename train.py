
import data_setup , engine  , utils
from model import UNet
from utils import create_writer , save_loss_curves , calculate_class_weights , count_class_occurrences

import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn

import argparse
from typing import Dict, List

#Set seeds for reproducibillity:
utils.set_seeds()


# # Setup hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

NUM_CLASSES = 9
FIRST_FEATURE_NUM = 64
NUM_LAYERS = 4

NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True

LOAD_MODEL = True

#Setup directories:
LABEL_PATH_TRAIN = 'dataset_p/patched_aerial_data/label/train_patches'
LABEL_PATH_VAL = 'dataset_p/patched_aerial_data/label/val_patches'

IMAGES_PATH_TRAIN = 'dataset_p/patched_aerial_data/images/train_patches'
IMAGES_PATH_VAL =  'dataset_p/patched_aerial_data/images/val_patches'

SAVE_CURVE = True


#Create transforms (* create transforms.py and import it)
DATA_TRANSFORM = transforms.Compose([
    transforms.PILToTensor()
])


# Setup device agnostic code:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


DEFAULT_CLASS_WEIGHTS = True


def main(args) :

  print(f"Learning rate: {args.LEARNING_RATE}")
  print(f"Batch size: {args.BATCH_SIZE}")
  print(f"Number of epochs: {args.NUM_EPOCHS}")
  print(f"Number of classes: {args.NUM_CLASSES}")
  print(f"First feature value: {args.FIRST_FEATURE_NUM}")
  print(f"Num layers: {args.NUM_LAYERS}")
  print(f"Device: {args.DEVICE}")
  print(f"Do load model: {args.LOAD_MODEL}")

  print(f'Num workers: {args.NUM_WORKERS}')
  print(f'Pin memory: {args.PIN_MEMORY}')

  print(f'Train labels path: {args.LABEL_PATH_TRAIN}')
  print(f'Train images path: {args.IMAGES_PATH_TRAIN}')
  print(f'Test labels path: {args.LABEL_PATH_VAL}')
  print(f'Test images path: {args.IMAGES_PATH_VAL}')

  print(f'Do save curve: {args.SAVE_CURVE}')
  print(f'Class weights: {args.DEFAULT_CLASS_WEIGHTS}')



  train_dataloader , test_dataloader , train_dataset , test_dataset = data_setup.create_dataloaders(train_labels_path = args.LABEL_PATH_TRAIN ,
                                                                                                    train_images_path = args.IMAGES_PATH_TRAIN ,
                                                                                                    test_labels_path = args.LABEL_PATH_VAL,
                                                                                                    test_images_path = args.IMAGES_PATH_VAL,
                                                                                                    transform = DATA_TRANSFORM ,
                                                                                                    batch_size = args.BATCH_SIZE ,
                                                                                                    pin_memory = args.PIN_MEMORY ,
                                                                                                    num_workers = args.NUM_WORKERS)


  # Create a model:
  model = UNet(in_channels=3, num_classes=args.NUM_CLASSES, first_feature_num=args.FIRST_FEATURE_NUM, num_layers=args.NUM_LAYERS).to(args.DEVICE)


  #Setup loss and optimizer:

  if args.DEFAULT_CLASS_WEIGHTS :
    class_percentage = count_class_occurrences(mask_dir = args.LABEL_PATH_TRAIN )
    CLASS_WEIGHTS = calculate_class_weights(class_percentage)

    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(args.DEVICE))
  else:
    loss_fn = nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(lr = args.LEARNING_RATE , params = model.parameters() )

  model_name = 'UNet'

  dataloader_name = 'non_overlapping_patches'

  # Train model:
  model_results = engine.train(model = model ,
                        train_dataloader =  train_dataloader ,
                        test_dataloader = test_dataloader ,
                        loss_fn = loss_fn ,
                        optimizer = optimizer ,
                        epochs = args.NUM_EPOCHS ,
                        device = args.DEVICE ,
                        model_name= model_name ,
                        writer = create_writer(experiment_name = dataloader_name ,
                                    model_name = model_name ,
                                    extra = f'{args.NUM_EPOCHS}_epochs'))

  if args.SAVE_CURVE :
    save_loss_curves(results= model_results , plot_path = f'result_graph/{dataloader_name}_{model_name}_{args.NUM_EPOCHS}_loss_curve.png')

if __name__ == '__main__' :

  parser = argparse.ArgumentParser(description="Train a model with given hyperparameters.")


  #Add all hyperparameters:
  parser.add_argument("--LEARNING_RATE", type=float, default=0.0001, help="Learning rate for the optimizer.")
  parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Batch size for training.")
  parser.add_argument("--NUM_EPOCHS", type=int, default=20, help="Number of epochs to train.")

  parser.add_argument("--NUM_CLASSES", type=int, default= 9, help="Number of classes existing in data.")
  parser.add_argument("--FIRST_FEATURE_NUM", type=int, default=64, help="First feature value for model.")
  parser.add_argument("--NUM_LAYERS", type=int, default=4, help="Number of layers in model.")
  parser.add_argument("--DEVICE", type= str, default= 'cpu', help="Device use for training: e.g. 'cpu' or 'cuda' ")
  parser.add_argument("--LOAD_MODEL", action='store_true', help="If set, You load/overwrite model checkpoint and continue training.")


  parser.add_argument("--NUM_WORKERS", type=int, default=NUM_WORKERS, help="Number of subprocesses to use for data loading.")
  parser.add_argument("--PIN_MEMORY", action='store_true', help="If set, pin memory for faster computation.")


  parser.add_argument("--LABEL_PATH_TRAIN", type=str, default= LABEL_PATH_TRAIN, help="Path for directory with train labels.")
  parser.add_argument("--IMAGES_PATH_TRAIN", type=str, default= IMAGES_PATH_TRAIN , help="Path for directory with train images.")
  parser.add_argument("--LABEL_PATH_VAL", type=str, default= LABEL_PATH_VAL, help="Path for directory with validation labels.")
  parser.add_argument("--IMAGES_PATH_VAL", type=str, default=IMAGES_PATH_VAL, help="Path for directory with validation images.")

  parser.add_argument("--SAVE_CURVE", action='store_true', help="If set, save loss curve after training.")

  parser.add_argument('--DEFAULT_CLASS_WEIGHTS', action= 'store_true' , help="If set, use balanced weights.")




  args = parser.parse_args()

  args.DEVICE = torch.device(args.DEVICE)

  main(args)



