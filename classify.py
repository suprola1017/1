import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser
from Multimodal_Emotion_Recognizer.ConvNeXt.convNeXt import ConvNeXt, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from sklearn.model_selection import train_test_split






if __name__ == "__main__":
        
        parser = ArgumentParser()

         # Arguments users used when running command lines
        parser.add_argument('--model', default='normal', type=str,
                        help='Type of ConvNeXt model, valid option: tiny, small, base, large, xlarge')
        parser.add_argument('--lr', default=0.0001,
                        type=float, help='Learning rate default=0.0001')
        parser.add_argument('--weight-decay', default=1e-3,
                                type=float, help='Weight decay default=1e-3')
        parser.add_argument("--batch-size", default=16, type=int)
        parser.add_argument("--epochs", default=30, type=int)
        parser.add_argument('--num-classes', default=3,
                                type=int, help='Number of classes')
        parser.add_argument('--image-height', default=32,
                                type=int, help='Size of input image')
        parser.add_argument('--image-width', default=32,
                                type=int, help='Size of input image')
        parser.add_argument('--image-channels', default=1,
                                type=int, help='Number channel of input image')
        parser.add_argument('--train-folder', default='', type=str,
                                help='Where training data is located')
        parser.add_argument('--valid-folder', default='', type=str,
                                help='Where validation data is located')
        parser.add_argument('--class-mode', default='sparse',
                                type=str, help='Class mode to compile')
        parser.add_argument('--model-folder', default='ERsavelog/',
                                type=str, help='Folder to save trained model')

        args = parser.parse_args()

        # Project Description
        for i, arg in enumerate(vars(args)):
                print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
        print('===========================')

        # Assign arguments to variables to avoid repetition
        train_folder = args.train_folder
        valid_folder = args.valid_folder
        batch_size = args.batch_size
        image_height = args.image_height
        image_width = args.image_width
        image_channels = args.image_channels
        num_classes = args.num_classes
        epochs = args.epochs
        class_mode = args.class_mode
        lr = args.lr
        weight_decay = args.weight_decay
        model_folder = args.model_folder
        
        # Load data
        train_data = np.load('A_Multi_Fusion_Data\merged_signal_train.npy')
        train_label = np.load('A_Multi_Fusion_Data\labels_train.npy')
        
        test_data = np.load('A_Multi_Fusion_Data\merged_signal_test.npy')
        test_label = np.load('A_Multi_Fusion_Data\labels_test.npy')

        # Transform labels
        ys_train_orig = np.zeros((train_label.shape[0],), dtype=int)
        for i in range((len(train_label))):
                train_label[i] = train_label[i] - 1
                ys_train_orig[i] = train_label[i][0].astype(int)
        Y_train = ys_train_orig
        train_data = np.expand_dims(train_data, axis=1)
        flag_train = np.zeros((train_data.shape[0], 1, 32, 32))
        for i in range(train_data.shape[0]):
                for j in range(32):
                        for k in range(32):
                                flag_train[i][0][j][k] = np.log(np.abs(train_data[i][0][j][k]))
        scaled_data = (flag_train - np.min(flag_train)) / (np.max(flag_train) - np.min(flag_train))  # 归一化到0-1范围
        flag_train = 2 * scaled_data - 1  # 将数据映射到[-1, 1]的范围
        X_train = flag_train
        
        
        ys_test_orig = np.zeros((test_label.shape[0],), dtype=int)
        for i in range((len(test_label))):
                test_label[i] = test_label[i] - 1
                ys_test_orig[i] = test_label[i][0].astype(int)
        Y_test = ys_test_orig
        test_data = np.expand_dims(test_data, axis=1)
        flag_test = np.zeros((test_data.shape[0], 1, 32, 32))
        for i in range(test_data.shape[0]):
                for j in range(32):
                        for k in range(32):
                                flag_test[i][0][j][k] = np.log(np.abs(test_data[i][0][j][k]))
        scaled_data = (flag_test - np.min(flag_test)) / (np.max(flag_test) - np.min(flag_test))  # 归一化到0-1范围
        flag_test = 2 * scaled_data - 1  # 将数据映射到[-1, 1]的范围
        X_test = flag_test
        
        #添加生成数据到原始训练集中
        GAN_Data = np.load('After_Gan_Data\GAN_DATA.npy')
        GAN_Label = np.load('After_Gan_Data\GAN_label.npy')
        
        X_train = np.row_stack([X_train, GAN_Data])
        Y_train = Y_train.reshape(-1, 1)
        Y_train = np.row_stack([Y_train, GAN_Label])
        Y_train = Y_train.reshape(-1,)
        
        # Print information about data shapes
        print()
        print("number of training examples =", X_train.shape[0])
        print("number of test examples =", X_test.shape[0])
        print("X_train shape:", X_train.shape)
        print("Y_train shape:", Y_train.shape)
        print("X_test shape:", X_test.shape)
        print("Y_test shape:", Y_test.shape)
        print()

        # Define the ConvNeXt model
         # ConvNeXt
        if args.model == 'tiny':
                model = convnext_tiny()
        elif args.model == 'small':
                model = convnext_small()
        elif args.model == 'base':
                model = convnext_base()
        elif args.model == 'large':
                model = convnext_large()
        elif args.model == 'xlarge':
                model = convnext_xlarge()
        else:
                model = ConvNeXt(
                num_classes=num_classes,
                in_channels=image_channels,
                )
                
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print("Using device:", device)
        
        # Move the model to GPU
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Convert NumPy arrays to PyTorch tensors
        X_train_tensor = torch.Tensor(X_train)
        Y_train_tensor = torch.LongTensor(Y_train)

        X_test_tensor = torch.Tensor(X_test)
        Y_test_tensor = torch.LongTensor(Y_test)

        # Create PyTorch data loaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Validation dataset and loader
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        
        #train process
        def save_checkpoint(model, epoch, model_folder):
                if not os.path.exists(model_folder):
                        os.makedirs(model_folder)
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(raw_model.state_dict(), model_folder+f"/epoch-{epoch+1}"+".pt")
                print("Model Saved")

        def load_checkpoint(self, path):
                self.model.load_state_dict(torch.load(path))
                print("Model loaded from", path)        
        
        # Training loop
        def run_epoch(split, loader):
            is_train = split == "train"
            if is_train:
                model.train()
            else:
                model.eval()

            losses = []
            accuracies = []
            correct = 0
            num_smamples = 0

            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                x,y = batch
                num_smamples += x.size(0)

                x = x.to(device)
                y = y.to(device)

                logits, loss = model(x, y)

                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1) 
                    correct += predictions.eq(y).sum().item()
                    accuracy = correct/num_smamples
                    accuracies.append(accuracy)
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f"Epoch: {epoch+1} it: {it+1} | loss: {loss.item():.5f} accuracy: {accuracy:.5f}")
                
            if not is_train:
                test_loss = float(np.mean(losses))
                test_accuracy = float(np.mean(accuracies))
                print(f"Test loss: {test_loss} accuracy: {test_accuracy}")
                return test_loss

        best_loss = float('inf')
        test_loss = float('inf')

        for epoch in range(epochs):
            run_epoch('train', train_loader)

            if test_dataset is not None:
                test_loss = run_epoch('test', test_loader)

            good_model = test_dataset is None or test_loss < best_loss
            if args.model_folder is not None and good_model:
                best_loss = test_loss
                save_checkpoint(model, epoch, model_folder)
