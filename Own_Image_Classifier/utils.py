import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from PIL import Image

class Util(object):

    @staticmethod
    def load_data(data_dir="./flowers" ):
        data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])
        
        transforms_test = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])
        
        transforms_val = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])
        
        # TODO: Load the datasets with ImageFolder
        # image_datasets = 
        
        dataset_train = datasets.ImageFolder(train_dir, transform=transforms_train)
        dataset_test = datasets.ImageFolder(test_dir, transform=transforms_test)
        dataset_val = datasets.ImageFolder(valid_dir, transform=transforms_val)
        
        # TODO: Using the image datasets and the trainforms, define the dataloaders
        # dataloaders = 
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=True)


        return dataloader_train , dataloader_val, dataloader_test, dataset_train

    @staticmethod
    def model_setup(architecure='vgg16', learning_rate=0.001, hardware='gpu'):

        architecures = { "vgg16":25088,
                        "resnet50":2048,
                        "alexnet":9216 }
        
        if architecure == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif architecure == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif architecure == 'alexnet':
            model = models.alexnet(pretrained = True)
        else:
            print("Please enter a valid architecure")

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        # Creating a feed forward network
        classifier = nn.Sequential(nn.Linear(architecures[architecure], 1588),
                                         nn.ReLU(),
                                         nn.Linear(1588, 488),
                                         nn.ReLU(),                                 
                                         nn.Linear(488, 102), 
                                         nn.LogSoftmax(dim=1))
        
        model.classifier = classifier

        # Use GPU if available else use cpu
        if torch.cuda.is_available() and hardware == 'gpu':
            model.cuda()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)



        return model, criterion, optimizer

    @staticmethod
    def testing_acc_check(model,dataloader_test,hardware='gpu'):    
        result_correct = 0
        result_total = 0
        
        with torch.no_grad():
            for data in dataloader_test:
                inputs, labels = data
                if torch.cuda.is_available() and hardware == 'gpu':
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                result_total += labels.size(0)
                result_correct += (predicted == labels).sum().item()
                
        print(f"testing_accuracy on 10000 test images: {(100 * result_correct / result_total)}")


    @staticmethod
    def train_network(dataloader_train, dataloader_test,model, criterion, optimizer, num_epochs=3, print_instance=10, hardware='gpu'):
        num_epochs = 3
        print_instance = 10
        
        for e in range(num_epochs):
            loss_running = 0
            stps = 0
            
            start = time.time()
            
            model.train()
            for inputs, labels in dataloader_train:
                stps += 1
                print("Started")
                optimizer.zero_grad()
                
                if torch.cuda.is_available() and hardware == 'gpu':
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                output = model.forward(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                loss_running += loss.item()
                
                if stps % print_instance == 0:
                    # Set network in evaluation mode for inference
                    model.eval()
                    print("Still running")
                    # Turn off gradients for validation to save memory and computations
                    with torch.no_grad():
                        loss_test = 0
                        acc = 0
                        for inputs, labels in dataloader_test:
                            if torch.cuda.is_available() and hardware == 'gpu':
                                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                            output = model.forward(inputs)
                            loss_test += criterion(output, labels).item()
                            ps = torch.exp(output)
                            equality = (labels.data == ps.max(dim=1)[1])
                            acc += equality.type(torch.FloatTensor).mean()
                        
                    print("Epoch: {}/{}.. ".format(e+1, num_epochs),
                          "Training Loss: {:.3f}.. ".format(loss_running/print_instance),
                          "Testing Loss: {:.3f}.. ".format(loss_test/len(dataloader_test)),
                          "Testing Accuracy: {:.3f}".format(acc/len(dataloader_test)))
                    
                    loss_running = 0
                    start = time.time()
                    
                    # Turn training back on
                    model.train()
                    
        print("Training Finished!")

    @staticmethod
    def save_checkpoint(model, dataset_train, path='checkpoint.pth'):
        model.class_to_idx = dataset_train.class_to_idx
        torch.save(model, path)

    @staticmethod
    def load_checkpoint(path='checkpoint.pth'):
        model = torch.load(path)
        model.eval()
        
        return model

    @staticmethod
    def process_image(img_path):
        
        img = Image.open(img_path)
    
         # Process the image
        preprocessing = transforms.Compose([
             transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], 
                                  [0.229, 0.224, 0.225])
         ])
    
        return preprocessing(img)

    
    @staticmethod
    def predict(img_pth, model, topk=5, hardware='gpu'):
        if torch.cuda.is_available() and hardware == 'gpu':
            model.to('cuda:0')

        img_torch = Util.process_image(img_pth)
        img_torch = img_torch.unsqueeze_(0).float()
        
        if hardware == 'gpu':
            with torch.no_grad():
                output = model.forward(img_torch.cuda())
        else:
            with torch.no_grad():
                output=model.forward(img_torch)
            
        probs = F.softmax(output.data,dim=1)
        
        return probs.topk(topk)
