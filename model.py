import segmentation_models_pytorch as smp
import torch



class MultiviewNet(torch.nn.Module):

    #torch.set_default_dtype(torch.half)

    def __init__(self, name_second_network, ch_input):

        super().__init__()

        channels1 = 6
        channels2 = ch_input #9

        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(ch_input, channels1, 
                            kernel_size = (3,3), 
                            stride = (1,1), 
                            padding = (1,1), 
                            bias = True),
            torch.nn.BatchNorm2d(channels1),
            torch.nn.ReLU()
        )
        self.conv_2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels1, channels2, 
                            kernel_size = (3,3), 
                            stride = (1,1), 
                            padding = (1,1), 
                            bias = True),
            torch.nn.BatchNorm2d(channels2),
            torch.nn.ReLU()
        )

        aux_params=dict(pooling = 'avg',             # one of 'avg', 'max'
                        dropout = 0.1,               # dropout ratio, default is None     # activation function, default is None
                        classes= 2,                 # define number of output labels
        )

        self.model = smp.Unet(encoder_name = name_second_network,
                              encoder_weights = None, #'noisy-student',
                              in_channels = channels2,
                              classes = 2,
                              aux_params = aux_params,
                              )
        

    def forward(self, x1):

        #output_after_conv_1_1 = self.conv_1_1(x1)
        #output_after_conv_2_1 = self.conv_2_1(output_after_conv_1_1)
        output, labels_emb = self.model(x1)
        #embeddings = self.model.encoder(output_after_conv_2_1)

        return output, labels_emb#embeddings,

    def freeze_decoder(self):

        for child in self.model.decoder.children():
            for param in child.parameters():
                param.requires_grad = False

        return

    def unfreeze(self):

        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = True

        return

