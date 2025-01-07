import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda")
print(device)


class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.iloc[idx, 1]
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 128)

        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pooling_layer(x)

        x = self.relu(self.conv2(x))
        x = self.pooling_layer(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        x = self.output_layer(x)

        return x



# Parameters to check
batch_sizes = [16, 32, 64]
learning_rates = [0.0005, 0.001, 0.01, 0.05, 0.1]
weight_decays = [1e-5, 1e-4, 1e-3]


# batch_sizes = [64]
# learning_rates = [0.001]
# weight_decays = [1e-3]

results = []
last_epoch = 0

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:

            seed = 110

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            train_df = pd.read_csv('dataset/train.csv')
            test_df = pd.read_csv('dataset/test.csv')

            class_names = train_df['label'].unique()
            class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

            train_df['label'] = train_df['label'].map(class_to_idx)

            print(f"Testing for batch size = {batch_size}, learning rate = {learning_rate}, weight_decay = {weight_decay}")

            train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed)
            train_dataset = ImageDataset(dataframe=train_df, img_dir='dataset/train', transform=transform)
            val_dataset = ImageDataset(dataframe=val_df, img_dir='dataset/train', transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))

            test_df = pd.read_csv('dataset/test.csv')
            test_dataset = ImageDataset(dataframe=test_df, img_dir='dataset/test', transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))


            def initialize_models(num_classes):
                models_dict = {
                    "custom_model": {
                        "model": CustomModel(num_classes),
                        "optimizer": None,
                        "name": 'CustomModel'
                    }
                    # "resnet50": {
                    #     "model": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
                    #     "optimizer": None,
                    #     "name": 'ResNet50'
                    # },
                    # "vgg11": {
                    #     "model": models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1),
                    #     "optimizer": None,
                    #     "name": 'VGG11'
                    # },
                    # "googlenet": {
                    #     "model": models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1),
                    #     "optimizer": None,
                    #     "name": 'GoogLeNet'
                    # },
                    # "alexnet": {
                    #     "model": models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
                    #     "optimizer": None,
                    #     "name": 'AlexNet'
                    # },
                    # "densenet121": {
                    #     "model": models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
                    #     "optimizer": None,
                    #     "name": 'DenseNet121'
                    # }
                }

                for model_key, model_info in models_dict.items():
                    model = model_info["model"]

                    if model_key == "resnet50" or model_key == "googlenet":
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                    elif model_key == "vgg11" or model_key == "alexnet":
                        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                    elif model_key == "densenet121":
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                    elif model_key == "custom_model":
                        model.output_layer = nn.Linear(model.output_layer.in_features, num_classes)

                    if model_key != "custom_model":
                        for param in model.parameters():
                            param.requires_grad = False

                        if model_key == "resnet50" or model_key == "googlenet":
                            for param in model.fc.parameters():
                                param.requires_grad = True
                        elif model_key == "vgg11" or model_key == "alexnet":
                            for param in model.classifier[6].parameters():
                                param.requires_grad = True
                        elif model_key == "densenet121":
                            for param in model.classifier.parameters():
                                param.requires_grad = True

                    models_dict[model_key]["model"] = model.to(device)
                    models_dict[model_key]["optimizer"] = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                return models_dict


            def train_model(model, train_loader, criterion, optimizer):
                model.train()
                running_loss = 0.0
                correct_preds = 0
                total_preds = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_preds += torch.sum(preds == labels).item()
                    total_preds += labels.size(0)

                epoch_loss = running_loss / len(train_loader)
                epoch_acc = correct_preds / total_preds
                return epoch_loss, epoch_acc


            def evaluate_model(model, val_loader, criterion):
                model.eval()
                running_loss = 0.0
                correct_preds = 0
                total_preds = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        correct_preds += torch.sum(preds == labels).item()
                        total_preds += labels.size(0)

                epoch_loss = running_loss / len(val_loader)
                epoch_acc = correct_preds / total_preds
                return epoch_loss, epoch_acc


            # Training parameters
            num_classes = len(class_names)
            models_dict = initialize_models(num_classes)
            criterion = nn.CrossEntropyLoss()
            best_val_acc_dict = {model_key: 0.0 for model_key in models_dict.keys()}
            num_epochs = 40

            # Early stopping parameters
            patience = 3
            counter_without_improvement = {model_key: 0 for model_key in models_dict.keys()}
            completed_training_flag = {model_key: False for model_key in models_dict.keys()}

            if not os.path.exists('Saved_states/'):
                os.makedirs('Saved_states/')

            for epoch in range(num_epochs):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('-' * 60)

                for model_key, model_info in models_dict.items():
                    model = model_info["model"]
                    optimizer = model_info["optimizer"]
                    model_name = model_info["name"]

                    if completed_training_flag[model_key] == False:
                        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
                        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

                        print(f'{model_name} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                        print(f'{model_name} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

                        if val_acc > best_val_acc_dict[model_key]:
                            best_val_acc_dict[model_key] = val_acc
                            torch.save(model.state_dict(), f'Saved_states/best_{model_key}_model.pth')
                            # print(f"{model_name} - model saved with improved validation accuracy.")
                            counter_without_improvement[model_key] = 0
                        else:
                            counter_without_improvement[model_key] += 1
                            if counter_without_improvement[model_key] >= patience:
                                print(f"{model_name} model has not improved for {patience} epochs. Early stopping.")
                                result_val_acc = best_val_acc_dict[model_key]
                                last_epoch = epoch
                                completed_training_flag[model_key] = True
                if all(completed_training_flag[model_key] for model_key in models_dict):
                    break

            print(f"Results for batch size = {batch_size} learning rate = {learning_rate}, weight_decay = {weight_decay}")
            print(result_val_acc)
            results.append([batch_size, learning_rate, weight_decay, result_val_acc, last_epoch])

results_df = pd.DataFrame(results, columns=["batch_size", "learning_rate", "weight_decay", "result_val_acc", "last_epoch"])
results_df = results_df.sort_values(by="result_val_acc", ascending=False)
results_df.to_csv('results_hyperparameters.csv', index=False)

print("Results saved to 'results_hyperparameters.csv'")
