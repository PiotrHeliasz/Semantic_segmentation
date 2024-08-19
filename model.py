
import torch
import torchvision.transforms as transforms
import torch.nn as nn


##########    DOUBLE CONV   #########

class DoubleConv(nn.Module) :
  '''Double convolution block with ReLU activation,
  presented at both sides of UNet.
  Added BatchNorm2d.
  '''
  def __init__(self, in_channels: int , out_channels: int) :
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels , out_channels , kernel_size = 3 , padding = 1 , bias = False) ,
        nn.BatchNorm2d(out_channels) ,
        nn.ReLU(inplace = True) ,

        nn.Conv2d(out_channels , out_channels , kernel_size = 3 , padding = 1 , bias = False ) ,
        nn.BatchNorm2d(out_channels) ,
        nn.ReLU(inplace = True)     )

  def forward(self , x : torch.tensor) :
    return self.conv(x)

###############    CONCATENATION   ###############

class CropCat(nn.Module):

  def forward(self, x: torch.tensor, contracting_x: torch.tensor):

    contracting_x = transforms.functional.center_crop(contracting_x, [x.shape[2] , x.shape[3] ])
    x = torch.cat([contracting_x , x ], dim=1)

    return x

###############    UNET   ###############

class UNet(nn.Module) :
  def __init__(self , in_channels = 3 , num_classes = 9 ,
               first_feature_num:int = 64 , num_layers:int = 4  ) :
    '''Architecture based on: https://github.com/aladdinpersson implementation of
    [U-Net: Convolutional Networks for Biomedical Image Segmentation].

    Args:

      in_channels (int): CC, by default in_channels = 3.
      num_classes (int): Number of segmented classes. By default num_classes = 9.
      first_feature_num (int): First feature determine model width.
      Highest feature number is equal to first_feature_num * 2**num_layers.
      num_layers (int): Determined model depth.
      You should always give input which is divisible by 2**num_layers.
      By default num_layers = 4, so input size should be divisible by 2**4 = 16.
    '''

    super().__init__()

    self.in_channels = in_channels
    self.num_classes = num_classes
    self.features = [first_feature_num * 2**i  for i in range(num_layers)  ] #we're gonna extract twice as many features each times as we goes down


    self.downs = nn.ModuleList()
    self.ups = nn.ModuleList()

    self.concat = CropCat()


    # Downsampling:
    for feature in self.features :  # [64, maxpool, 128, maxpool , 256 , maxpool , 512 , maxpool]
      self.downs.append(DoubleConv(in_channels , feature) )
      in_channels = feature
      self.downs.append(nn.MaxPool2d(kernel_size = 2 , stride = 2 ) )

    # Upsampling:
    for feature in reversed(self.features):
        self.ups.append( nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)  )  # DOuble the height and the width
        self.ups.append(DoubleConv(feature * 2, feature))


    # Bottleneck:
    self.bottleneck = DoubleConv(self.features[-1] , self.features[-1] * 2 ) # By Default: [512 , 512 * 2 = 1024 ]

    # Final layer:
    self.final_conv = nn.Conv2d(self.features[0] , self.num_classes , kernel_size = 1 )


  def forward(self , x : torch.tensor) :
    skip_connections = []

    #Downsampling:
    for idx in range(0 , len(self.downs) , 2) :
      x = self.downs[idx](x)     # DoubleConv   0,2,4,6,8
      skip_connections.append(x)
      x = self.downs[idx +1](x)  # maxpool

    #Bottleneck:
    x = self.bottleneck(x)

    #Upsampling:
    skip_connections = skip_connections[:: -1 ] #reverse list to move backward through features

     #Every even idx is convTranspose, and odd is Double_conv:
    for idx in range(0 , len(self.ups) , 2 ) :
      x = self.ups[idx](x)
      skip_connection = skip_connections[idx//2]


      x = self.concat(skip_connection , x   )

      x = self.ups[idx+1](x)


    # Final layer:
    return self.final_conv(x)




