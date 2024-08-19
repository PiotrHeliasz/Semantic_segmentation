
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

from utils import save_model , save_checkpoint , load_checkpoint , create_writer

from tqdm.auto import tqdm
from typing import Dict , List , Tuple

# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime


#Write device agnostic code:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

def train(model: torch.nn.Module ,
          train_dataloader : torch.utils.data.DataLoader ,
          test_dataloader : torch.utils.data.DataLoader ,
          optimizer : torch.optim.Optimizer ,
          writer : torch.utils.tensorboard.writer.SummaryWriter ,
          loss_fn : torch.nn.Module = nn.CrossEntropyLoss() ,
          epochs : int = 5 ,
          device: torch.device = device ,
          load_model: bool = True  ,
          model_name: str = 'new_model.pth' ,
          checkpoint_path: str = 'checkpoint.pth'  ) -> Dict[str, List] :

  ''' Complete function for training.
  Args:
    model (torch.nn.Module): Model to train.
    train_dataloader (torch.utils.data.DataLoader): Training dataloader.
    test_dataloader (torch.utils.data.DataLoader): Validating/testing dataloader.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    writer (torch.utils.tensorboard.writer.SummaryWriter):
    SummaryWriter() instance tracking to a specyfic directory.
    loss_fn (torch.nn.Module): Loss function for training, by default
    loss_fn = nn.CrossEntropyLoss().
    epochs (int): Number of epochs during training.
    device (torch.device): Device used for training.
    load_model (bool): By default load_model = True, every 3 epochs,
    model state_dict, optimizer, and epoch will be saved.
    model_name (str): Filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension. By default,
    model_name = 'new_model.pth' is saved in: 'models/new_model.pth'
    checkpoint_path (str): Path to checkpoint which you want to load,
    to continue training a model.
    By default checkpoint_path = 'checkpoint.pth'.
    'checkpoint.pth' is the path where checkpoint will overwrite itself.

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=3:
             {train_loss: [2.0616, 1.0537 , 0.8543],
              train_acc: [0.256, 0.432, 0.5213],
              test_loss: [1.2641, 1.5706 , 1.001],
              test_acc: [0.22, 0.2973 , 0.4323]}
  '''


  # Create empty results dictionary:
  results = {'train_loss' : [] ,
             'train_acc' : [] ,
             'test_loss' : [] ,
             'test_acc' : [] }

  checkpoint_path = f'{model_name}_{checkpoint_path}'

  start_epoch = 0
  if load_model :
    start_epoch = load_checkpoint(model , optimizer , filename = checkpoint_path )

  assert start_epoch < epochs , f'You cannot continue training from {start_epoch} epoch to {epochs} epoch. Start epoch should be smaller than epochs!'

  #Loop through training and testing loops for a number of epochs:
  for epoch in tqdm(range(start_epoch , epochs ) ) :

    # Put model in train mode:
    model.train()

    # Setup evaluation metrics: train/test loss and train/test accuracy values:
    test_loss = test_acc = train_loss = train_acc = 0

    for batch , (img , mask) in enumerate(train_dataloader  ) :
      #Send data to correct devices:
      img , mask = img.to(device) , mask.to(device)

      #Forward pass:
      logits = model(img)

      #Calc. Loss and pixel accuracy:
      loss = loss_fn(logits , mask.argmax(dim=1) )
      train_loss += loss.item()

      y_pred = logits.argmax(dim = 1)
      # print(f'Shape of y_pred: {y_pred.shape}\nLength of y_pred: {len(y_pred)}')
      train_acc += (y_pred == mask.argmax(dim=1)).sum().item() / ( y_pred.numel() )

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    #Adjast metrics to get average loss and accuracy per batch:
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    #Put model in evaluation mode:
    model.eval()

    #Turn on inference mode:
    with torch.inference_mode() :
      # Loop through DataLoader:
      for batch, (img , mask) in enumerate(test_dataloader) :
        #Send data to correct device:
        img , mask = img.to(device) , mask.to(device)

        #Forward pass:
        logits = model(img)

        #Cals. loss and accuracy:
        loss = loss_fn(logits , mask )
        test_loss += loss.item()

        y_pred = logits.argmax(dim = 1)
        test_acc += (y_pred == mask.argmax(dim=1)).sum().item() / ( y_pred.numel() )  #len(y_pred)


      #Adjust metrics to get average loss and accuracy per batch:
      test_loss /= len(test_dataloader)
      test_acc /= len(test_dataloader)


    ### PRINTS AND CHECKPOINTS:
    print(f'Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {(100*train_acc):.4f}% | Test loss: {test_loss:.4f} | Test acc: {(100*test_acc):.4f}%')

    if (epoch) % 4 == 0   :  #  to start from next one epoch
      save_checkpoint(model, optimizer, epoch , filename=f'{model_name}_checkpoint.pth' )


    # Update results dictionary:
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

    ### Experiment tracking:
    if writer :

      writer.add_scalars(main_tag = 'Loss' ,
                        tag_scalar_dict = {'train_loss': train_loss ,
                                            'test_loss' : test_loss} ,
                        global_step = epoch )
      writer.add_scalars(main_tag = 'Accuracy' ,
                        tag_scalar_dict = {'train_acc' : train_acc ,
                                            'test_acc' : test_acc} ,
                        global_step = epoch )


      #Close the writer:
      writer.close()

    else :
      pass

  save_checkpoint(model, optimizer, epoch , filename=f'{model_name}_checkpoint.pth' )

  # Save the model state dict:
  save_model(model=model,
            target_path= 'models',
            model_name= f'{model_name}_{epochs}_epochs.pth')

  # Return the filled results at the end of the epoch:
  return results

