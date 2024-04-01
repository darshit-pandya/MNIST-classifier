import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)



def reduce_dim(queries, n):
    # Using list comprehension to create chunks and compute their averages
    reduced_list = [sum(queries[i:i + n]) / len(queries[i:i + n]) for i in range(0, len(queries), n) if len(queries[i:i + n]) > 0]
    return reduced_list



def read_mnist(file_name):

    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)

def show_mnist(file_name,mode):

    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

class NN(nn.Module):
    def __init__(self, input_size, drop):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.reg = nn.Dropout(drop)     # question 3

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.reg(x)   # question 3
        x = F.relu(self.fc2(x))
        x = self.reg(x)     # question 3
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MNISTDataset(Dataset):
    def __init__(self, data, reduction_step):
        self.data = data
        self.reduction_step = reduction_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, pixels = self.data[idx]
        pixels = np.array(pixels, dtype=np.float32)

        # Apply dimensionality reduction
        reduced_pixels = reduce_dim(pixels, self.reduction_step)

        # Normalize the reduced pixels
        reduced_pixels = np.array(reduced_pixels) / 255.0
        return torch.tensor(reduced_pixels, dtype=torch.float), torch.tensor(int(label))


def calculate_accuracy(model, data_loader):
  def accuracy_calc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
   






def classify_mnist(drop, weight_decay,  reduction_step=2):
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')

    batch_size = 64
    train_dataset = MNISTDataset(train, reduction_step)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #print(train_loader)

    valid_dataset = MNISTDataset(valid, reduction_step)
    validation_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = MNISTDataset(test, reduction_step)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 784 // reduction_step
    model = NN(input_size, drop)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    # if op == "Adam":
    #     optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    # elif op =="RMSprop":
    #     optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)
    # elif op =="Adagrad":
    #     optimizer = optim.Adagrad(model.parameters(), weight_decay=weight_decay)


    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    num_epochs = 10  # Adjust this as per your requirement
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.squeeze()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(total_train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        all_preds_valid = []
        all_labels_valid = []
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

                all_preds_valid.extend(predicted.numpy())
                all_labels_valid.extend(labels.numpy())

        valid_losses.append(total_valid_loss / len(validation_loader))
        valid_accuracies.append(100 * correct_valid / total_valid)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Train Accuracy: {train_accuracies[-1]:.2f}%, '
              f'Validation Loss: {valid_losses[-1]:.4f}, '
              f'Validation Accuracy: {valid_accuracies[-1]:.2f}%')

    # Evaluate on the test set
    model.eval()
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds_test.extend(predicted.numpy())
            all_labels_test.extend(labels.numpy())

        correct_test = np.sum(np.array(all_preds_test) == np.array(all_labels_test))
        total_test = len(all_labels_test)

        test_accuracy = (correct_test / total_test) * 100
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        f1_sklearn = f1_score(all_labels_test, all_preds_test, average='weighted')
        print(f'F1 Score: {f1_sklearn:.2f}')

    # Plotting learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Confusion Matrix for Test Set
    cm_test = confusion_matrix(all_labels_test, all_preds_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.show()

    
# checking different activation functions 
# print('Leaky ReLu:')
# print('Final Model')
# classify_mnist(0,0) 

# checking for best optimizer
# print('Adam:')
# classify_mnist(0,0,"Adam") 
# print('Adagrad:')
# classify_mnist(0,0,"Adagrad") 
# print('RMSprop:')
# classify_mnist(0,0,"RMSprop") 




# op = "Adam"
# act = "ReLu"
# Question 2 - No regularization 
#print('Without Regularization:')
#classify_mnist(0,0,op)   # parameter 1 - dropout, parameter 2 - weight_decay



# regularization

print('Without Regularization:')
classify_mnist(0,0)

print('\n With 50% Dropout regularization:')
classify_mnist(0.50,0)   # parameter 1 - dropout, parameter 2 - weight_decay



# print('\n With 75% Dropout regularization:')
# classify_mnist(0.75,0)   # parameter 1 - dropout, parameter 2 - weight_decay

# print('\n With Weight Decay regularization:')
# classify_mnist(0,0.001)   # parameter 1 - dropout, parameter 2 - weight_decay
