from torchsummary import summary 
import models.resnet
import torchvision.models

model = models.resnet.resnet50(num_classes=100)
# model = torchvision.models.resnet50(num_classes=100)
summary(model, (3, 32, 32))
