import torch
import numpy as np
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset



class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset


    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()

        # loader = torch.utils.data.DataLoader(dataset=dataset,
        #                                       batch_size=self.batch,
        #                                       shuffle=self.shuf,
        #                                       num_workers=0,
        #                                       drop_last=True)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------准备数据阶段-------------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------------------------------------------------------------#

        #加载数据集
        train_data = np.load('/E22201085/Liao/Self-Attention-GAN-master/datasets/3_WESAD_NO_EDA_merged_signal.npy')
        train_label = np.load('/E22201085/Liao/Self-Attention-GAN-master/datasets/3_WESAD_NO_EDA_labels.npy')

         #将WESAD标签中全部减一
        ys_train_orig = np.zeros((train_label.shape[0],), dtype=int)
        for i in range((len(train_label))):
            train_label[i] = train_label[i] - 1
            ys_train_orig[i] = train_label[i][0].astype(int)
     

        #将data，label换个名字
        X_train, Y_train = train_data, ys_train_orig
            
        #扩充X的维度
        X_train = np.expand_dims(X_train, axis=1)

        #_____________________________________________________________________________________________________________
        flag = np.zeros((X_train.shape[0], 1, 116, 30))
        for i in range(X_train.shape[0]):
            for j in range(116):
                for k in range(30):
                    flag[i][0][j][k] = np.log(np.abs(X_train[i][0][j][k]))

        #将数据缩放到[-1, 1]的范围
        scaled_data = (flag - np.min(flag)) / (np.max(flag) - np.min(flag))  # 归一化到0-1范围
        flag = 2 * scaled_data - 1  # 将数据映射到[-1, 1]的范围
        
        X_train = flag
       # _____________________________________________________________________________________________________________

        print(X_train)
        print("number of training examples = " + str(X_train.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))

        # 转换数据为张量
        X_train_tensor = torch.from_numpy(X_train).float()
        Y_train_tensor = torch.from_numpy(Y_train).long()

        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=self.shuf)
        return loader

