import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, Linear
from tqdm.auto import tqdm
import sys
from torchvision.models import resnet18

from models import *
from pre_process import *
from visualize_plots import *


def get_y_pred(output):
    y_pred = []
    for out in output:
        if out > 0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = torch.tensor(y_pred).to(device)
    return y_pred


def training(model, optimizer, criterion):
    model.train()
    epoch_train_accuracy = 0
    epoch_train_loss = 0
    for image, label in train_dl:
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        y_pred = get_y_pred(output)
        #print(output, y_pred, label)
        loss = criterion(output.squeeze(), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_train_accuracy += len(torch.where(y_pred == label)[0]) / (len(train_dl))
        epoch_train_loss += loss.item() / len(train_dl)
        # print(epoch_train_accuracy, epoch_train_loss)

    return epoch_train_accuracy, epoch_train_loss


def validation(model, criterion):
    model.eval()
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    with torch.no_grad():
        for image, label in val_dl:
            image = image.to(device)
            label = label.to(device)

            val_output = model(image)
            y_pred = get_y_pred(val_output)
            # val_loss = criterion(val_output, label)
            val_loss = criterion(val_output.squeeze(), label)

            # acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += len(torch.where(y_pred == label)[0]) / len(val_dl)
            epoch_val_loss += val_loss / len(val_dl)

    return epoch_val_accuracy, epoch_val_loss


def testing(model, criterion):
    model.eval()
    acc = []
    loss = []
    with torch.no_grad():
        for image, label in test_dl:
            image = image.to(device)
            label = label.to(device)

            test_output = model(image)
            y_pred = get_y_pred(test_output)
            loss.append(criterion(test_output.squeeze(), label))
            acc.append(len(torch.where(y_pred == label)[0]))
    return acc, loss


def get_acc_loss_and_plot(model_name, model, optimizer):
    # Define loss criterion: Binary Cross Entropy Loss with Logits
    criterion = BCEWithLogitsLoss()

    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        # Train model
        acc, loss = training(model, optimizer, criterion)
        train_accuracy.append(acc)
        train_loss.append(loss)
        # if epoch % 10 == 0:
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, acc, loss))

        # Validate results
        acc, loss = validation(model, criterion)
        val_accuracy.append(acc)
        val_loss.append(loss)
        # if epoch % 10 == 0:
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, acc, loss))

    # Test model on unseen data
    acc, loss = testing(model, criterion)
    test_acc = acc
    test_loss = loss
    print('Average test_accuracy : {}, test_loss : {}'.format(
        sum(test_acc) / len(test_acc), sum(test_loss) / len(test_loss)))

    # Plot and save training and validation loss and accuracy
    visualize(model_name, train_loss, train_accuracy, val_loss, val_accuracy)


##############################################################################

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(10)
if device == 'cuda':
    torch.cuda.manual_seed_all(10)
# Number of classes in our dataset
num_classes = 1

# Definition of hyperparameters for CNN model
batch_size = 10
num_epochs = 20
lr = 0.001

# Get train, val, test data loaders
train_dl, val_dl, test_dl = data_loaders(batch_size)

# Store all print statements in a txt file
# sys.stdout = open("all_print_stmt.txt", "w")

print("CNN will be trained")
# Get training, val, testing loss and accuracy and their plots for both models
# Create CNN model
cnn = CNNModel(num_classes)
cnn.to(device)
print(cnn)
# Define optimizer for cnn model
optimizer = optim.Adam(cnn.parameters(), lr=lr)
# Get loss and accuracy plots for train, val, test sets
get_acc_loss_and_plot("cnn", cnn, optimizer)

###############################################################################

print("Modified ResNet-18 will be trained")

# Definition of hyperparameters for CNN model
num_epochs = 5
lr = 0.01

# Load the modified Resnet-18 pre-trained model
modified_resnet18 = resnet18(pretrained=True)
# Freeze all except last linear layer for fine tuning during training
for param in modified_resnet18.parameters():
    param.requires_grad = False
modified_resnet18.fc = Linear(in_features=512, out_features=num_classes, bias=True)  # Modified last layer
modified_resnet18.to(device)
# Define optimizer
optimizer = optim.Adam(modified_resnet18.parameters(), lr=lr)
# Get loss and accuracy plots for train, val, test sets
get_acc_loss_and_plot("resnet", modified_resnet18, optimizer)

# sys.stdout.close()
