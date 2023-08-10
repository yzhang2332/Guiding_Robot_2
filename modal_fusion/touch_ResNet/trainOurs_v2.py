import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from dataset_v2 import MyDataset  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import numpy as np
from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import resnet34Ours
import time

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     ]),  # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   ])}    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    image_path = 'act_data_v2'  # data set path !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.DatasetFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_dataset = MyDataset(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    validate_dataset = MyDataset(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    
    net = resnet34Ours()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet34-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 9)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # net.load_state_dict(torch.load('./resNet34_4_4.pth'))  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 200
    best_acc = 0.0
    save_path = './resNet34_v2.pth'  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_steps = len(train_loader)

    x_index_data = []
    y_train_loss = []
    y_valid_accu = []

    time2 = time.time()
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            time1 = time.time()
            # print('time1:', time1-time2)

            images, labels = data
            images = images.float()
            optimizer.zero_grad()
            logits = net(images.to(device))
            labels = labels.type(torch.LongTensor)  # 从torch.int32转换为64,避免报错expected scalar type Long but found Int
            loss = loss_function(logits, labels.to(device))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            time2 = time.time()
            # print('time2:', time2 - time1)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        y_true = list()
        y_pred = list()
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.float() 

                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                y_true.append(val_labels)
                y_pred.append(predict_y.detach().cpu().numpy())

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %(epoch + 1, running_loss / train_steps, val_accurate))
        x_index_data.append(epoch + 1)
        y_train_loss.append(running_loss / train_steps)
        y_valid_accu.append(val_accurate)

        f = open("cls_traindata.txt", "w")
        f.write(str(x_index_data)+"\n")
        f.write(str(y_train_loss)+"\n")
        f.write(str(y_valid_accu)+"\n")
        f.close()

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

            y_true = np.concatenate(y_true).reshape(-1)
            y_pred = np.concatenate(y_pred).reshape(-1)
            cm = confusion_matrix(y_true, y_pred)
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
            plot_confusion_matrix(cm, classes, './confusion_v2.png', title='confusion matrix')  # !!!!!!!!!!!!!!!!!
        print("epoch: ", epoch, "; best_acc: ", best_acc)

    print(cm)
    print('Finished Training')


if __name__ == '__main__':
    main()
