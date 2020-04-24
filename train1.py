from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import copy
import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
class MyDataset(Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        fh = os.listdir(path)
        imgs = []
        self.path=path
        for image in fh:
            imgs.append(path+'/'+image)
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index],int(self.imgs[index][len(self.path)+1])
        img = Image.open(fn) #.convert('1')
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        return len(self.imgs)


train_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Lambda(lambda img: _crop(img,224))
])

path_train = './formal_train_set'
path_test = './formal_test_set'
train_data = MyDataset(path_train,train_transforms)
test_data = MyDataset(path_test,train_transforms)
train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=50,shuffle=True,
       num_workers=0)
    #   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),pin_memory=True,
test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=100,shuffle=False,
       num_workers=0)
       
# /***********************************************************************/
label_every=torch.zeros((7,7))
# for data in train_data:
#     label_every[data[1]]+=1
# print(label_every)
    
# /***************************************************/

dataloaders={}
dataset_sizes={}
dataloaders['train']=train_queue
dataloaders['val']=test_queue
dataset_sizes['train']=len(train_data)
dataset_sizes['val']=len(test_data)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
     
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            la=copy.deepcopy(label_every)
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for i in range(len(preds)):
                    la[labels.data[i],preds[i]]+=1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(la)
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# def visualize_model(model, num_images=6):
#     images_so_far = 0
#     fig = plt.figure()
#     for i, data in enumerate(dataloaders['val']):
#         inputs, labels = data
#         inputs, labels = Variable(inputs), Variable(labels)
#         outputs = model(inputs)
#         _, preds = torch.max(outputs.data, 1)
#         for j in range(inputs.size()[0]):
#             images_so_far += 1
#             ax = plt.subplot(num_images//2, 2, images_so_far)
#             ax.axis('off')
#             ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#             if images_so_far == num_images:
#                 return

# model_conv = models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 7)
# model=model_conv.cuda()

# criterion = nn.CrossEntropyLoss()
# criterion = criterion.cuda()

# # Observe that only parameters of final layer are being optimized as
# # opoosed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
torch.cuda.set_device(0)
cudnn.benchmark = True
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 7)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
model = model_ft.cuda()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print(model_ft)
#model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
# Finetuning the convnet