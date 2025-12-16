"""
This code is partly taken from one of the exercises of course DATA.ML.200,
but heavily modified to fit the purpose of this project.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchaudio
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

path = "processed_data"

# A helper class to load the audio data
# Assumes that training data is under path/train, testing data under path/test
# Both are devided into car_wav and tram_wav directories that have the wav files
class DataGenerator(Dataset):
    def __init__(self, mode, verbose=False):
        super(DataGenerator, self).__init__()

        self.duration_per_file_in_s = 5  # crop the input audios to 5 seconds
        self.sampling_rate = 16000
        self.samples_per_file = int(self.duration_per_file_in_s * self.sampling_rate)

        self.car_files = glob.glob(os.path.join(path, mode, 'car_wav', '*.wav'))
        self.tram_files = glob.glob(os.path.join(path, mode, 'tram_wav', '*.wav'))
        if verbose: print(mode, "cars:", len(self.car_files))
        if verbose: print(mode, "trams:", len(self.tram_files))
        if verbose: print()

        # Combine all files for indexing
        self.files = self.car_files + self.tram_files
        self.labels = [0] * len(self.car_files) + [1] * len(self.tram_files)

    def __getitem__(self, item):
        file_path = self.files[item]
        label = self.labels[item]

        audio_data, sr = torchaudio.load(file_path)  # returns a tensor with shape [channels, samples]

        if audio_data.size(1) > self.samples_per_file:
            audio_data = audio_data[:, :self.samples_per_file]  # Truncate
        else:
            pad_length = self.samples_per_file - audio_data.size(1)
            audio_data = torch.nn.functional.pad(audio_data, (0, pad_length))  # Pad at the end

        return audio_data, label

    def __len__(self):
        return len(self.files)


# Helper class for simplifying the construction of the model
# One block consists of a convolution layer, batchnormalization, tanh activation and dropout layers
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation_fn = nn.Tanh()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x


# Construct a CNN with four convolution layers (BasicBlocks)
# After them, some pooling, flattening and Sigmoid as activation
class MyModel(nn.Module):
    def __init__(self, kernel_size=37, stride=7, conv_channels=(32,16,8,1)):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(conv_channels)):  # four basic blocks
            in_channels = 1 if i == 0 else conv_channels[i - 1]
            out_channels = conv_channels[i]
            block = BasicBlock(in_channels, out_channels, kernel_size, stride)
            self.layers.append(block)

        self.feature_extractor = nn.Sequential(*self.layers)

        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = self.flatten(x)
        x = torch.sigmoid(x)

        return x


class CNNClassifier:
    def __init__(self, kernel_size=37, stride=7, conv_channels=(32,16,8,1), verbose=False):
        # Select the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose: print("Device:", self.device)

        # Some optimization for cuda
        torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler('cuda')

        # Loading the data
        train_data = DataGenerator(mode='train')
        test_data = DataGenerator(mode='test')
        self.train_dataloader = DataLoader(train_data, batch_size=8, pin_memory=True, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=8, pin_memory=True, shuffle=True)

        # Creating the model
        self.model = MyModel(kernel_size=kernel_size, stride=stride, conv_channels=conv_channels).to(self.device)
        torch.compile(model=self.model, mode='reduce-overhead')

        # Initializing criterion and optimizer
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Load the model if one exists
        if os.path.exists("CNNclassifier.pth"):
            self.model.load_state_dict(torch.load('CNNclassifier.pth', map_location=self.device))
            if verbose: print("Loaded weights from your saved model successfully!")

    # Training (around ~50s depending on luck)
    def fit(self, epochs=100, threshold=0.88, verbose=False, graphs=False):
        self.model.train()
        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        # Train for multiple epochs
        for epoch in range(epochs):
            start = time.time()
            total_loss, correct_predictions = 0., 0.

            # Training in multiple batches
            for i, (input_batch, target_batch) in enumerate(self.train_dataloader):
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device).float()
                self.optimizer.zero_grad(set_to_none=True)

                # Utilize cuda AMP and scaler for speed
                with torch.amp.autocast('cuda'):
                    predictions = self.model(input_batch).flatten()
                    loss = self.criterion(predictions, target_batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                correct_predictions += ((predictions.detach() >= 0.5).int() == target_batch.int()).sum().item()

            # Calculate, print, and log accuracies and losses
            train_loss = total_loss / (i+1)  # i+1 = num of bathces
            train_acc = correct_predictions / len(self.train_dataloader.dataset)
            test_loss, test_acc, _, _ = self.eval()
            if verbose: print(f'Epoch {epoch}, train_loss {train_loss:.2f}, train_acc {train_acc:.4f}, test_loss {test_loss:.2f}, test_acc {test_acc:.4f}, epoch time {time.time() - start:.2f} s')
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(test_loss)
            val_accs.append(test_acc)

            if test_acc > threshold:  # The model is good enough
                torch.save(self.model.state_dict(), 'CNNclassifier.pth')
                if verbose: print("Your trained model is saved successfully!")
                break

        if graphs:
            epochs = range(len(train_losses))  # num of executed epochs

            plt.plot(epochs, train_losses, label="Train loss")
            plt.plot(epochs, val_losses, label="Validation loss")
            plt.legend()
            plt.title("Loss")
            plt.show()

            plt.plot(epochs, train_accs, label="Train accuracy")
            plt.plot(epochs, val_accs, label="Validation accuracy")
            plt.legend()
            plt.title("Accuracy")
            plt.show()

        return train_losses, train_accs, val_losses, val_accs

    # Evaluating (took ~0.38s for whole test-set)
    # returns the loss, accuracy, precision, and recall on the test material
    def eval(self):
        self.model.eval()
        with (torch.no_grad()):
            total_loss = 0.
            preds = torch.zeros(len(self.test_dataloader.dataset)).to(self.device)
            targets = torch.zeros(len(self.test_dataloader.dataset)).to(self.device)
            n = 0
            for i, (input_batch, target_batch) in enumerate(self.test_dataloader):
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.float().to(self.device)

                predictions = self.model(input_batch).flatten()
                loss = self.criterion(predictions, target_batch)

                total_loss += loss.item()
                bsz = predictions.size(0)  # batch size
                preds[n:n+bsz] = predictions
                targets[n:n+bsz] = target_batch
                n += bsz

        targets = targets.cpu().numpy()
        preds = preds.cpu().numpy().round()

        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)

        return total_loss/(i+1), accuracy, precision, recall


if __name__ == '__main__':
    classifier = CNNClassifier()
    l, a, p, r = classifier.eval()
    print("Loss:", l, ", Accuracy:", a, ", Precision:", p, ", Recall:", r)
