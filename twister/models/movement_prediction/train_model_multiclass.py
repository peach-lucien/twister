import matplotlib.pyplot as plt
import pandas as pd

import time

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelBinarizer


from twister.models.utils import seed_everything, construct_data_dictionary
from twister.models.cnn_model import dataset, model, fit, validate

# model directory should have a train and validation folder and label dictionary (see construct_dataset.py)
data_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/model_datasets/movement_model_dataset/'
output_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/models/movement_models/'


#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                           construct data dictionary
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#

# getting label dictionary
movement_labels = pd.read_csv(data_directory +'label_dict.csv', index_col = 0)

# defining column from label dictionary
label_id = 'movement'

# define directories for storing data for model
train_directory = data_directory + 'train_data/'
val_directory = data_directory + 'val_data/'

# construct data dictionaries for reading from pytorch
#train_data = construct_data_dictionary(train_directory, movement_labels, label_id)
#val_data = construct_data_dictionary(val_directory, movement_labels, label_id)



#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                           parameters and initialisation
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#

# define parameters
height=540 # height of images
width=540 # width of images
epochs = 30 # number of epochs for learning
SEED=42 # random seed
batch_size = 32 # batch size for learning

# set computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clearing memory
torch.cuda.empty_cache()

# set random seed
seed_everything(SEED=SEED)

#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                  loading data and constructing data loaders
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#

# loading label dictionary
labels = pd.read_csv(data_directory +'label_dict.csv', sep=',',index_col = 0)

# load train data 
df = pd.read_csv(data_directory + 'train_data/data.csv')
xtrain = df.image_path.values
ytrain = df.target.values   

# load val data 
df = pd.read_csv(data_directory + 'val_data/data.csv')
xtest = df.image_path.values
ytest = df.target.values   


# construct dataset objects
train_data = dataset(xtrain, ytrain, tfms=1, dtype=torch.long)
test_data = dataset(xtest, ytest, tfms=0, dtype=torch.long)
 
# initiate dataloaders
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                              building CNN model 
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#

n_output = labels.shape[0]

# import model
cnn = model(pretrained=True, requires_grad=True, n_output=n_output)
cnn = torch.nn.DataParallel(cnn)
cnn = cnn.to(device)

# define optimizer
optimizer = torch.optim.SGD(cnn.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.0005)

# define loss function
criterion = torch.nn.CrossEntropyLoss()

#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                       training and evaluating model 
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#


# empty lists for collecting data
train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []


start = time.time() # setting timer
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    
    # train model
    train_epoch_loss, train_epoch_accuracy = fit(cnn, trainloader, criterion, optimizer, device, train_data)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)    
    
    # evaluate model
    val_epoch_loss, val_epoch_accuracy = validate(cnn, testloader, criterion, device, test_data)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)    


end = time.time() # ending timer  

#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                       plotting and saving model
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#



# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(output_directory +'accuracy_multilabel.png')
plt.show()


# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(output_directory +'loss_multilabel.png')
plt.show()


# save the model to disk
print('Saving model...')
torch.save(cnn.state_dict(), output_directory +'model_multilabel.pth')
