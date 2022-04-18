import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets

import time
import matplotlib.pyplot as plt

def train(model, train_loader, test_loader, device):
    
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                                     milestones=[3,5])
    
    training_losses = []
    training_acc = []
    training_time = []
    test_losses = []
    test_acc = []
    test_time = []

    for epoch in range(50):

      # train
      model.train()
      losses = 0.0
      
      correct = 0
      total = 0

      start_time = time.time()
      for _, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward + backward + optimizer + scheduler
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses += loss.item()

        _, predictions = outputs.max(1)
        correct += (predictions == labels).sum()
        total += predictions.size(0)
      
      tr_time = time.time() - start_time
      training_time.append(tr_time)
      avg_train_loss = losses / len(train_loader)
      training_losses.append(avg_train_loss)
      tr_acc = float(correct) / float(total) * 100
      training_acc.append(tr_acc)
      
      print(f"Epoch: {epoch}")
      print("Average training loss: %.4f, Accuracy : %.2f%%, Time(s): %.2f"
            % (avg_train_loss, tr_acc, tr_time))

        
      # evaluate on the 10,000 test images
      model.eval()
      losses = 0.0
      
      with torch.no_grad():
        correct = 0
        total = 0

        start_time = time.time()
        for _, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            losses += loss.item()

            _, predictions = outputs.max(1)
            correct += (predictions == labels).sum()
            total += predictions.size(0)

        te_time = time.time() - start_time
        test_time.append(te_time)
        avg_test_loss = losses / len(test_loader)
        test_losses.append(avg_test_loss)
        te_acc = float(correct) / float(total) * 100
        test_acc.append(te_acc)

        print("Average test loss: %.4f, Accuracy : %.2f%%, Time(s): %.2f"
              % (avg_test_loss, te_acc, te_time))

    # plot
    epochs = [i for i in range(1, 51)]
    plt.plot(epochs, training_losses, "mediumseagreen", label="training_losses")
    plt.plot(epochs, test_losses, "mediumpurple", label="test_losses")
    plt.xticks(epochs)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("losses.png")
    plt.show()

    plt.plot(epochs, training_acc, "mediumseagreen", label="training_accuracy")
    plt.plot(epochs, test_acc, "mediumpurple", label="test_accuracy")
    plt.xticks(epochs)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("accuracy.png")
    plt.show()

    plt.plot(epochs, training_time, "mediumseagreen", label="training_time")
    plt.plot(epochs, test_time, "mediumpurple", label="test_time")
    plt.xticks(epochs)
    plt.xlabel("epochs")
    plt.ylabel("time")
    plt.savefig("time.png")
    plt.show()


def main():
    # load CIFAR-10 dataset
    transform_train = transforms.Compose([transforms.Resize((227,227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    batch_size = 64

    train_data = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    
    # train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # AlexNet(Normal)
    model1 = AlexNet()
    model1.to(device)
    train(model1, train_loader, test_loader, device)

    # AlexNet(SeparableConv2D)
    model2 = AlexNet_S()
    model2.to(device)
    train(model2, train_loader, test_loader, device)

if __name__ == '__main__':
    main()