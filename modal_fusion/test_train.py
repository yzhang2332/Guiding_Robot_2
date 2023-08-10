import torch
import numpy as np
import multi_swin
from datasets import DogDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Args:
    def __init__(self) -> None:
        self.epochs, self.learning_rate, self.patience = [30, 0.001, 4]
        self.batch_size = 1
        self.lr = 0.001
        self.epochs = 1000
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.logs = "modal_fusion/logs/train_log"
        self.video_frames = 30


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score+self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss


args = Args()
writer = SummaryWriter(log_dir=args.logs,flush_secs=60)

test_dataset = DogDataset(annotations_file='modal_fusion/datasets/test_data/processed_data/annotations.csv',
                          audio_dir="modal_fusion/datasets/test_data/processed_data/audio",
                          video_dir="modal_fusion/datasets/test_data/processed_data/video",
                          touch_dir="modal_fusion/datasets/test_data/processed_data/touch",
                          pose_dir="modal_fusion/datasets/test_data/processed_data/pose"
                          )
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=0)

# ======================train============================

model = multi_swin.MultiSwin(
    input_channels=1, output_classes=20,final_output_classes=17).to(args.device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

early_stopping = EarlyStopping(patience=args.patience, verbose=True)

for epoch in range(args.epochs):
    model.train()
    train_epoch_loss = []
    for idx, sample_batch in enumerate(test_dataloader, 0):
        # audi
        # print(sample_batch['audio'].shape, sample_batch['video'].shape,
        #       sample_batch['touch'].shape, sample_batch['pose'].shape)
        
        # print(sample_batch['video'].shape)
        # video_split = torch.split(sample_batch['video'], args.video_frames, dim = 2)

        # for i in range(args.video_frames):
        #     # print(video_batch.shape)
        #     if i == 0:
        #         video_batch =video_split[0]
        #     else:
        #         video_batch = torch.stack((video_batch,video_split[i]),dim=0)
        #         print(int((i)*sample_batch['video'].size(2)/args.video_frames))
            # print(video_batch.shape)
        # video_batch = torch.permute(video_batch, (1, 0, 2, 3))
        # print(sample_batch['video'].shape)
        outputs = model(sample_batch['audio'].unsqueeze(1).to(args.device), sample_batch['video'].to(args.device), sample_batch['touch'].to(args.device), sample_batch['pose'].unsqueeze(1).to(args.device))
        optimizer.zero_grad()
        loss = criterion(outputs, sample_batch['label'].to(args.device))
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        # if idx % (len(test_dataloader)//2) == 0:
        print("--epoch={}/{}    {}/{}of train     loss={}".format(
                epoch, args.epochs, idx, len(test_dataloader), loss.item()))
        writer.add_video('video', sample_batch['video'], epoch*len(test_dataloader)+idx)
        writer.add_images('touch', sample_batch['touch'], epoch*len(test_dataloader)+idx, dataformats='NCHW')
        writer.add_scalar('train_loss', loss.item(), epoch*len(test_dataloader)+idx)
    train_epochs_loss.append(np.average(train_epoch_loss))

    # # =====================valid============================
    # with torch.no_grad():
    #     model.eval()
    #     valid_epoch_loss = []
    #     for idx, sample_batch in enumerate(test_dataloader, 0):
    #         outputs = model(sample_batch['audio'].unsqueeze(1).to(args.device), sample_batch['video'].to(args.device), sample_batch['touch'].to(args.device), sample_batch['pose'].unsqueeze(0).to(args.device))

    #         loss = criterion(outputs, sample_batch['label'].to(args.device))
    #         valid_epoch_loss.append(loss.item())
    #         valid_loss.append(loss.item())
    #         # if idx % (len(test_dataloader)//2) == 0:
    #         print("--epoch={}/{}    {}/{}of val     loss={}".format(
    #                 epoch, args.epochs, idx, len(test_dataloader), loss.item()))
    #         writer.add_video('video', sample_batch['video'], epoch*len(test_dataloader)+idx)
    #         writer.add_image('touch', sample_batch['touch'], epoch*len(test_dataloader)+idx, dataformats='NCHW')
    #         writer.add_scalar('valid_loss', loss.item(), epoch*len(test_dataloader)+idx)
    #     valid_epochs_loss.append(np.average(valid_epoch_loss))
    # # ==================early stopping======================
    # early_stopping(
    #     valid_epochs_loss[-1], model=model, path=r'./model')
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break
    # # ====================adjust lr========================
    lr_adjust = {
        2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        10: 5e-7, 15: 1e-7, 20: 5e-8
    }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

# =========================save model=====================
torch.save(model.state_dict(), 'model.pth')
writer.close()

# plt.figure(figsize=(12, 4))
# plt.subplot(121)
# plt.plot(train_loss[:])
# plt.title("train_loss")
# plt.subplot(122)
# plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
# plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
# plt.title("epochs_loss")
# plt.legend()
# plt.show()

