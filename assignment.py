import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from visualization import TrainingVisualizer
import time
import os
import random
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

os.makedirs("output_data", exist_ok=True)

# RUNTIME CONSTS
SEED = 250904

IMG_SIZE = 32
SM_BATCH_SIZE = 64
OUTPUTS = 10
EPOCHS = 20
SM_LEARNING_RATE = 0.001
NUM_CLASSES = 10
DROPOUT = 0.001

NUM_CHANNELS = 3 # rgb
SM_NUM_TRAIN_WORKERS = 4
SM_NUM_TEST_WORKERS = 2

C_IN_CHANNELS = 8
C_OUT_CHANNELS = 8

run_time = time.time()
DATETIME_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_FILE = os.path.join("output_data", f"out_{DATETIME_STR}.txt")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

data_test = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# same seed for all generators
rng = torch.Generator().manual_seed(SEED)
torch.cuda.manual_seed(seed=SEED)
torch.manual_seed(seed=SEED)
np.random.seed(seed=SEED)
random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# DONE - softmax portion model class
class SoftMax(nn.Module):
    def __init__(self, img_dim: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(img_dim * img_dim * 3, num_classes),
        )

    def forward(self, ins):
        out = self.flatten(ins)
        out = self.linear_stack(out)
        return out


# DONE - softmax trainer tester class
# class SoftMaxTest():
#     def __init__(self, model, output_file="training.log"):
#         self.model = model.to(device)
#         self.output_file = output_file

#     def seed_worker(self, worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)
#         return worker_seed

#     def dataLoader(self, dataset, batchSize, numWorkers, shuffle=False):
#         loader_args = {
#             "dataset": dataset,
#             "batch_size": batchSize,
#             "num_workers": numWorkers,
#             "worker_init_fn": self.seed_worker,
#             "generator": rng,
#             "persistent_workers": numWorkers > 0,
#         }
#         if shuffle:
#             loader_args["shuffle"] = True
#         return DataLoader(**loader_args)

#     def saveResults(self, values, training=False):
#         with open(self.output_file, "a") as f:
#             if training:
#                 epoch, step, loss = values
#                 f.write(f"Epoch {epoch + 1}, Step {step}, Loss: {loss:.4f}\n")
#             else:
#                 avg_loss, accuracy = values
#                 f.write(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%\n")

#     def getLoss(self, logits, lbls):
#         return nn.CrossEntropyLoss()(logits, lbls)

#     def getAccuracy(self, logits, lbls):
#         outputs = torch.softmax(logits, dim=1)
#         _, predicted = torch.max(outputs, 1)
#         return (predicted == lbls).sum().item()

#     def train(self, trainData, epochs, batchSize, numWorkers, optimizer, criterion):
#         self.model.train()
#         data_loader = self.dataLoader(trainData, batchSize, numWorkers, shuffle=True)

        # variable = None

        # for epoch in range(epochs):
        #     running_loss = 0.0
        #     for i, (inputs, lbls) in enumerate(data_loader, 1):
        #         inputs, lbls = inputs.to(device), lbls.to(device)
        #         optimizer.zero_grad()

#                 logits = self.model(inputs)
#                 loss = criterion(logits, lbls)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 if i % 100 == 0:
#                     self.saveResults([epoch, i, running_loss / 100], training=True)
#                     print(f"[{epoch + 1}, {i:5d}] loss: {running_loss / 100:.3f}")
#                     running_loss = 0.0

#         print("Finished Training")

#     def test(self, testData, batchSize, numWorkers, criterion=None):
#         self.model.eval()
#         if criterion is None:
#             criterion = nn.CrossEntropyLoss()

#         data_loader = self.dataLoader(testData, batchSize, numWorkers)
#         total_loss = 0.0
#         correct = 0
#         samples = 0

#         with torch.no_grad():
#             for inputs, lbls in data_loader:
#                 inputs, lbls = inputs.to(device), lbls.to(device)
#                 logits = self.model(inputs)
#                 total_loss += criterion(logits, lbls).item() * lbls.size(0)
#                 correct += self.getAccuracy(logits, lbls)
#                 samples += lbls.size(0)

#         loss_avg = total_loss / samples if samples else 0.0
#         accuracy = correct / samples if samples else 0.0
#         self.saveResults([loss_avg, accuracy], training=False)
#         print(f"Test Loss: {loss_avg:.4f} | Test Accuracy: {accuracy*100:.2f}%")
#         return loss_avg, accuracy


# DONE - CNN model portion class
class CNN(nn.Module):
    # DONE - initialize CNN and configure its layers
    def __init__(
            self, img_dim: int, num_classes: int, num_c_layers: int, c_blocksize: int, num_fc_layers: int, num_fc_neurons: int
            ):
        super().__init__()

        self.c_blocksize = c_blocksize
        self.img_dim = img_dim

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layers_c = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        self.createLayers(num_c_layers, c_blocksize, num_fc_layers, num_fc_neurons, num_classes)
    
    # DONE - build convolutional layer blocks
    def createCLayers(self, num_layers, blocksize):

        input_channels = 3
        output_channels = 8

        self.layers_c = nn.ModuleList()

        # first layer
        self.layers_c.append(nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        ))

        input_channels = output_channels

<<<<<<< HEAD
        for _ in range(num_layers):

            output_channels = input_channels * 2

            for __ in range(blocksize):

                layer = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU()
                )

                self.layers_c.append(layer)

                input_channels = output_channels  # 🔥 MUST be here
=======
        # Create num_layers additional layers
        for i in range(num_layers-1):
            # Double channels every blocksize layers
            if blocksize > 0 and i % blocksize == 0 and i > 0:
                output_channels = input_channels * 2
            
            self.layers_c.append(nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            ))
            
            input_channels = output_channels
        
        print(f"Total convolutional layers created: {len(self.layers_c)}")
>>>>>>> fd52104 (new working with changed train function)

    # DONE - build fully connected layer sequence
    def createFCLayers(self, num_classes, num_neurons, num_layers):
        new_img_size = self.fetchNewImgDim(self.img_dim)

        next_layer = nn.Sequential(
            nn.Linear(
                new_img_size,
                num_neurons
            ),
            nn.ReLU()
        )
        self.layers_fc.append(next_layer)
        
        for _ in range(num_layers):
            
            next_layer = nn.Sequential(
                nn.Linear(
                    num_neurons,
                    num_neurons
                ),
                nn.ReLU()
            )
            self.layers_fc.append(next_layer)
        
        next_layer = nn.Sequential(
            nn.Linear(
                num_neurons,
                num_classes
            )
        )
        self.layers_fc.append(next_layer)
        # test

    # DONE - assemble convolutional and fully connected layers
    def createLayers(self, num_c_layers, c_blocksize, num_fc_layers, num_fc_neurons, num_classes):
        self.createCLayers(num_c_layers, c_blocksize)
        self.createFCLayers(num_classes, num_fc_neurons, num_fc_layers)
<<<<<<< HEAD

        num_features = self.fetchNewImgDim(32)
        self.fc_norm = nn.LayerNorm(num_features)

    def createFCLayers(self, num_classes, num_neurons, num_layers):
        new_img_size = self.fetchNewImgDim(self.img_dim)

        next_layer = nn.Sequential(
        nn.Linear(new_img_size,num_neurons), nn.ReLU())
        self.layers_fc.append(next_layer)
        
        for _ in range(num_layers):
            
            next_layer = nn.Sequential(
                nn.Linear(num_neurons, num_neurons), nn.ReLU())
            self.layers_fc.append(next_layer)
        
        next_layer = nn.Sequential(
            nn.Linear(
                num_neurons,
                num_classes
            )
        )
        self.layers_fc.append(next_layer)
        # test
=======
>>>>>>> fd52104 (new working with changed train function)
    
    # DONE - compute flattened feature dimension for classifier input
    def fetchNewImgDim(self, img_dim):

        self.eval()
        with torch.no_grad():

            x = torch.zeros(1, 3, img_dim, img_dim)

            for i, layer in enumerate(self.layers_c):

                x = layer(x)

                if self.c_blocksize > 0 and (i + 1) % self.c_blocksize == 0:
                    x = self.pool(x)

            features = x.view(1, -1).shape[1]

        self.train()
        return features
    
    # DONE - forward pass through the CNN model
    def forward(self, x):
        for i, layer_c in enumerate(self.layers_c):
            x = layer_c(x)

            if (i + 1) % self.c_blocksize == 0:
                x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_norm(x)
        for fcLayer in self.layers_fc[:-1]:
            x = fcLayer(x)

        x = self.layers_fc[-1](x)
        return x


# DONE - helper class for training and testing the CNN
class CNNTrainer():
    # DONE - initialize with model and output file
    def __init__(self, model, output_file="training.log"):
        self.model = model.to(device)
        self.output_file = output_file

    # DONE - initialize deterministic worker rng
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        return worker_seed

    # DONE - create a pytorch dataloader
    def dataLoader(self, dataset, batchSize, numWorkers, shuffle=False):
        loader_args = {
            "dataset": dataset,
            "batch_size": batchSize,
            "num_workers": numWorkers,
            "worker_init_fn": self.seed_worker,
            "generator": rng,
            "persistent_workers": numWorkers > 0,
        }
        if shuffle:
            loader_args["shuffle"] = True
        return DataLoader(**loader_args)

    # DONE - append training or testing results
    def saveResults(self, values, training=False):
        with open(self.output_file, "a") as f:
            if training:
                epoch, step, loss = values
                f.write(f"Epoch {epoch + 1}, Step {step}, Loss: {loss:.4f}\n")
            else:
                avg_loss, accuracy = values
                f.write(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%\n")

    # DONE - calculate CEL
    def getLoss(self, logits, lbls):
        lossFunc = nn.CrossEntropyLoss()
        loss = lossFunc(logits, lbls)
        return loss

    # DONE - acc soft
    def getAccuracy(self, logits, lbls):
        _, predicted = torch.max(logits, 1)
        return (predicted == lbls).sum().item()

    # DONE - train the model over a dataset
<<<<<<< HEAD

    def train(self, trainData, epochs, batchSize, trainNumWorkers, optimizer, criterion):
        self.model.train()
        dataLoader = self.dataLoader(trainData, batchSize, trainNumWorkers)

        for epoch in range(epochs): 
=======
    def train(self, train_data, epochs, batch_size, num_workers, optimizer, criterion):
        self.model.train()
        data_loader = self.dataLoader(train_data, batch_size, num_workers, shuffle=True)

        for epoch in range(epochs):
>>>>>>> fd52104 (new working with changed train function)
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader, 1):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

<<<<<<< HEAD
                optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    data = [epoch, i+1, running_loss/100]
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')

=======
                logits = self.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    self.saveResults([epoch, i, running_loss / 100], training=True)
                    print(f"Epoch {(epoch + 1):2d}, Batch {i:6d}, Loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

        print("Training complete!")
>>>>>>> fd52104 (new working with changed train function)

    # DONE - evaluate the model on test data
    def test(self, test_data, batch_size, num_workers, criterion=None):
        self.model.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        data_loader = self.dataLoader(test_data, batch_size, num_workers)
        loss_total = 0.0
        correct = 0
        samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = self.model(inputs)
                loss_total += criterion(logits, labels).item() * labels.size(0)
                correct += self.getAccuracy(logits, labels)
                samples += labels.size(0)

        loss_avg = loss_total / samples if samples else 0.0
        accuracy = correct / samples if samples else 0.0
        self.saveResults([loss_avg, accuracy], training=False)
        print(f"Testing Loss: {loss_avg:.4f} | Testing Accuracy: {accuracy*100:.2f}%")
        return loss_avg, accuracy


if __name__ == "__main__":
    # SOFTMAX

    print("--- SoftMax and CNN model and testing ---\n")

    with open(OUTPUT_FILE, "a") as f:
        f.write(
            f"----- PARAMETERS -----\n" \
            f"EPOCHS           : {EPOCHS}\n" \
            f"SM_BATCH_SIZE    : {SM_BATCH_SIZE}\n" \
            f"DROPOUT          : {DROPOUT}\n" \
            f"SM_LEARNING_RATE : {SM_LEARNING_RATE}\n" \
            f"NUM_CLASSES      : {NUM_CLASSES}\n"
            f"----------------------\n\n"
        )
    
    print(" - Training SoftMax...\n")
    
    # softmax_model = SoftMax(IMG_SIZE, NUM_CLASSES)
    # softmax_test = SoftMaxTest(softmax_model)
    # softmax_optimiser = torch.optim.Adam(softmax_model.parameters(), SM_LEARNING_RATE)
    # softmax_criterion = nn.CrossEntropyLoss()

    # softmax_test.train(data_train, EPOCHS, SM_BATCH_SIZE, SM_NUM_TRAIN_WORKERS, softmax_optimiser, softmax_criterion)
    # sm_loss_test, sm_accuracy_test = softmax_test.test(data_test, SM_BATCH_SIZE, SM_NUM_TEST_WORKERS)

    out_buffer = f"----- SoftMax Training Results -----\n" \
            f"Epochs completed : {EPOCHS}\n\n" \
            f"Batch size       : {SM_BATCH_SIZE}\n" \
            f"Learning rate    : {SM_LEARNING_RATE}\n" \
            f"Dropout          : {DROPOUT}\n" \
            f"- COMPLETE -\n" \
            # f"Test loss        : {sm_loss_test}\n" \
            # f"Test accuracy    : {sm_accuracy_test} ({(sm_accuracy_test*100):.2f}%)\n" \

    with open(OUTPUT_FILE, "a") as f:
        f.write(
            out_buffer
        )
    
    print(out_buffer)

    # CNN 2, 8, 16 and 32 layers

    cn_learning_rate = SM_LEARNING_RATE
    cn_num_train_workers = 0
    cn_num_test_workers = 0
    cn_learning_rate = 0.01
    cn_batch_size = 128

    for i in range(1, 6):
        if i == 2: # skip i^2==4 layer
            continue
        num_c_layers = pow(2, i)
        c_blocksize = 2
        num_fc_layers = 2
        num_fc_neurons = 128

        with open(OUTPUT_FILE, "a") as f:
            f.write(
                f"----- PARAMETERS -----\n" \
                f"num_c_layers   : {num_c_layers}\n" \
                f"c_blocksize    : {c_blocksize}\n" \
                f"num_fc_layers  : {num_fc_layers}\n" \
                f"num_fc_neurons : {num_fc_neurons}\n" \
                f"----------------------"
            )
        
        print(f" - Training CNN... ({num_c_layers} layers)")

        cn_model = CNN(
            IMG_SIZE,
            NUM_CLASSES,
            num_c_layers,
            c_blocksize,
            num_fc_layers,
            num_fc_neurons
        )
<<<<<<< HEAD
=======
        cn_model.to(device)

        # cn_optimiser = optim.SGD(
        #     cn_model.parameters(), cn_learning_rate, momentum=0.75, weight_decay=0.001
        #     )

        cn_optimiser = optim.Adam(
            cn_model.parameters(), cn_learning_rate
        )
        
>>>>>>> fd52104 (new working with changed train function)
        cn_trainer = CNNTrainer(cn_model)
        cn_criterion = nn.CrossEntropyLoss()
        cn_optimiser = optim.SGD(
            cn_model.parameters(), cn_learning_rate, momentum=0.9, weight_decay=0.0001
            )
        
        cn_trainer.train(
            data_train,
            EPOCHS,
            cn_batch_size,
            cn_num_train_workers,
            cn_optimiser,
            cn_criterion
        )

        cn_loss_test, cn_accuracy_test = cn_trainer.test(data_test, cn_batch_size, cn_num_test_workers)

        with open(OUTPUT_FILE, "a") as f:
            f.write(
                f"----- CNN training results -----\n" \
                f"Number of convolutional : {num_c_layers}\n" \
                f"          layers \n" \
                f"Epochs completed        : {EPOCHS}\n\n" \
                f"Batch size              : {cn_batch_size}\n" \
                f"Learning rate           : {cn_learning_rate}\n" \
                f"- COMPLETE -\n" \
                f"Test loss               : {cn_loss_test}\n" \
                f"Test accuracy           : {cn_accuracy_test} ({(cn_accuracy_test*100):.2f}%)\n" \
                f"-----------------------------------\n\n"
            )