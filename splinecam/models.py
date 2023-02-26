from torchvision import models
import torch.nn as nn

def vgg11_bn(n_class=200,input_res=64,weights=None):
    """
    Converts maxpools to avgpool and makes usable via library
    """
    
    if weights is None:
        weights = models.VGG11_BN_Weights.IMAGENET1K_V1
    
    if input_res == 64:
        
        model_ft = models.vgg11_bn(weights)

        model_ft.features[3] = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        model_ft.features[7] = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        model_ft.features[14] = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        model_ft.features[21] = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        model_ft.features[28] = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        model_ft.avgpool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        model_ft.classifier = nn.Linear(512,n_class)
        model_ft = nn.Sequential(*[each for each in model_ft.features],
#                                  model_ft.avgpool,
                                 nn.Flatten(),model_ft.classifier)
        
        # remove conv bias
        for each in [0,4,8,11,15,18,22,25]:
            model_ft[each].bias = None
        
        model_ft[0].stride=2

    else:
        raise NotImplemented
        
    return model_ft
    
    
def vgg16_bn(n_class=200, input_res=64, weights=None):
    
    if weights is None:
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1
        
    model = models.vgg16_bn(weights)
    model.features[0].stride=(2,2)

    layers = []
    for each in model.features:
        if type(each) == torch.nn.Conv2d:
            each.bias = None
            layers.append(each)

        elif type(each) == torch.nn.modules.MaxPool2d:
            layers.append(
                torch.nn.modules.AvgPool2d(
                    kernel_size=each.kernel_size,
                    padding=each.padding,
                    stride=each.stride,
                )
            )

        else:
            layers.append(each)

    model.features = torch.nn.Sequential(*layers)

    model.avgpool = torch.nn.Sequential()
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(512,2048),
        torch.nn.ReLU(inplace=True),
    #     torch.nn.Dropout(p=0.5),
        torch.nn.Linear(2048,512),
        torch.nn.ReLU(inplace=True),
    #     torch.nn.Dropout(p=0.5),
        torch.nn.Linear(512,200)
    )

    model = torch.nn.Sequential(
        *model.features,
        *model.classifier
    )
    
    return model
    
    