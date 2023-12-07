
import numpy as np

import random
from tqdm import tqdm

from PIL import Image
import albumentations

import torch
from torchvision import models
from torch.utils.data import Dataset
import torch.nn as nn


def model(pretrained, requires_grad, n_output):
    """  defining CNN model for learning  """
    
    # use resnet 50 pretrained model    
    model = models.resnet50(progress=True, pretrained=pretrained)
    
    # freeze hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
            
    # train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
            
    # construct final learnable output layer
    #model.fc = nn.Linear(2048, n_output)
    model.fc = nn.Sequential( nn.Dropout(0.5), nn.Linear(2048, n_output) )
                
    return model



class dataset(Dataset):
    """ dataset class for CNN """
    
    def __init__(self, path, labels, tfms=None, dtype=torch.long):
        
        self.X = path # path to images
        self.y = labels # ground truth labels
        self.dtype = dtype # dtype of input data
        
        # apply augmentations
        if tfms == 0: # if validating
            self.aug = albumentations.Compose([
                            # resize to 540 x 540
                            albumentations.Resize(540, 540, always_apply=True),
                            
                            # normalize image
                            albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225], always_apply=True)
                            ])
        else: # if training
            self.aug = albumentations.Compose([
                albumentations.Resize(540, 540, always_apply=True),   
                
                # shifting image with small rotations
                albumentations.ShiftScaleRotate(
                    shift_limit=0.3,
                    scale_limit=0.3,
                    rotate_limit=15,
                    p=0.5
                ),
                
                # crop and pad
                albumentations.CropAndPad(percent=0.1, p=0.5),
                
                # add color jitter
                albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                
                # sometimes distort
                albumentations.OneOf([
        		    albumentations.OpticalDistortion(p=0.3),
        		    albumentations.GridDistortion(p=.1),
        		], p=0.2),
                
                # sometimes change brightness
    		    albumentations.OneOf([
        		    albumentations.CLAHE(clip_limit=2),
        		    albumentations.RandomBrightnessContrast(),            
        		], p=0.3),
                
                # sometimes add gaussian noise
        		albumentations.OneOf([
        		    albumentations.GaussNoise(),
        		], p=0.2),
                
                # sometimes add blur
        		albumentations.OneOf([
        		    albumentations.MotionBlur(p=.2),
        		    albumentations.MedianBlur(blur_limit=3, p=0.1),
        		    albumentations.Blur(blur_limit=3, p=0.1),
        		], p=0.2),
                
                # add hue saturation
        		albumentations.HueSaturationValue(p=0.3),
                
                # sometimes blur
                albumentations.Blur(blur_limit=3),
                
                # always normalize
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225], always_apply=True)
                ])
            
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        """ outputting image and target label """
        image = Image.open(self.X[i])
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=self.dtype)



def compute_loss(outputs, target, criterion):
    """ calculate loss """    
   
    if isinstance(criterion, torch.nn.modules.loss.BCELoss):        
        loss = criterion(torch.sigmoid(outputs), target)   #! CHECK THIS            
    else:
        # standard loss function
        loss = criterion(outputs, target)  
        
    return loss

def compute_acc(outputs, target, criterion):
    """  calculate accuracy """     
    
    # accuracy for cross entropy loss
    if isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
        _, preds = torch.max(outputs.data, 1)    
        acc = (preds == target).sum().item() 
        
    # accuracy for binary cross entropy loss
    elif isinstance(criterion, torch.nn.modules.loss.BCELoss) or isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss)  or isinstance(criterion, torch.nn.modules.loss.MultiLabelSoftMarginLoss):
        preds = (torch.sigmoid(outputs) > 0.5).long()
        acc = (preds == target).all(1).sum().item() 
       
    # accuracy for mean square error loss
    elif isinstance(criterion, torch.nn.modules.loss.MSELoss):
        acc = ((outputs - target)*(outputs - target)).sum().item()

    return acc

#validation function
def validate(model, dataloader, criterion, device, test_data):
    """ Validating model """
    
    print('Validating')
    
    # set model to evaluation mode
    model.eval()
    
    # initiating running loss and accuracy
    running_loss = 0.0
    running_acc = 0
    
    # no gradient during validation
    with torch.no_grad():
        
        # looping over batches from dataloader
        for i, data in tqdm(enumerate(dataloader), total=int( len(test_data) / dataloader.batch_size )):
            
            # input images and labels
            data, target = data[0].to(device), data[1].to(device)
            
            # predict outputs for input images
            outputs = model(data)
            
            # compute batch loss 
            loss = compute_loss(outputs, target, criterion)
            running_loss += loss
            
            # compute batch accuracy
            running_acc += compute_acc(outputs, target, criterion)
            
    # calculate average loss 
    loss = running_loss / len(dataloader.dataset)
    
    # calculate total accuracy
    if type(criterion) is torch.nn.modules.loss.MSELoss: 
        accuracy = np.sqrt( running_acc / len(dataloader.dataset) )
    else:
        accuracy  = 100. * running_acc/len(dataloader.dataset)
    
    print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')
           
    return loss, accuracy


def fit(model, dataloader, criterion, optimizer, device, train_data):
    """ Training model """
    
    print('Training')
    
    # set model to training mode
    model.train()
    
    # initiating running loss and accuracy
    running_loss = 0.0
    running_acc = 0    
       
    # looping over batches from dataloader
    for i, data in tqdm(enumerate(dataloader), total=int( len(train_data) / dataloader.batch_size )):
        
        # input images and labels
        data, target = data[0].to(device), data[1].to(device)

        # set gradients to zero (accumulate gradients on every loss.backward call)
        optimizer.zero_grad()        

        # predict outputs for input images
        outputs = model(data)
        
        # compute batch loss 
        loss = compute_loss(outputs, target, criterion)
        running_loss += loss
        
        # compute batch accuracy
        running_acc += compute_acc(outputs, target, criterion)
        
        # accumulate (sum) the gradients
        loss.backward()
        
        # update gradient parameters 
        optimizer.step()
    
    # calculate average loss
    loss = running_loss / len(dataloader.dataset)

    # calculate total accuracy
    if type(criterion) is torch.nn.modules.loss.MSELoss: 
        accuracy = np.sqrt( running_acc / len(dataloader.dataset) )
    else:
        accuracy  = 100. * running_acc/len(dataloader.dataset)
    
    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")
           
    return loss, accuracy




def predict_label(image, model):
    """ predicting label for new image """ 
    
    # define resize and normalisation for image
    aug = albumentations.Compose([
                    albumentations.Resize(540, 540, always_apply=True),
                    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225], always_apply=True)
                ])  
    
    # apply augmentation
    image= aug(image = np.array(image))['image']    
    
    # reshape into correct format
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)    
    image = torch.tensor(image, dtype=torch.float).clone().detach().requires_grad_(False).unsqueeze(0)    
    
    # predict image
    out = model(image)   
    
    # softmax of output
    probs = nn.functional.softmax(out, dim=1).cpu().detach().numpy()[0]   
    
    # sigmoid of output
    pred = torch.sigmoid(out).cpu().detach().numpy()[0] 
    
    # raw output to numpy
    out = out.cpu().detach().numpy()[0] 
    
    return out, probs, pred


def load_trained_model(model_path, n_output):
    """ Loading a trained model """
    
    # set computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}")


    # instantiate model
    cnn_model = model(pretrained=False, requires_grad=False, n_output=n_output)
    cnn_model.eval()
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model = cnn_model.to(device)

    # load parameters
    if torch.cuda.is_available():
        model_dict = torch.load(model_path)
    else:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))

    cnn_model.load_state_dict(model_dict)
    
    return cnn_model











