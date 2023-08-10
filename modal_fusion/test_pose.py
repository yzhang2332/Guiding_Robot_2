import torch
import numpy as np
import multi_swin
from datasets import DogDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    '''Test the dataset with the sample data.'''

    test_dataset = DogDataset(annotations_file='modal_fusion/datasets/test_data/processed_data/annotations.csv',
                            audio_dir="modal_fusion/datasets/test_data/processed_data/audio",
                            video_dir="modal_fusion/datasets/test_data/processed_data/video",
                            touch_dir="modal_fusion/datasets/test_data/processed_data/touch",
                            pose_dir="modal_fusion/datasets/test_data/processed_data/pose"
                            )
    test_dataloader = DataLoader(
        test_dataset, batch_size= 1, shuffle=True, num_workers=0)
    
    model = multi_swin.MultiSwin(
    input_channels=1, output_classes=100, final_output_classes=17, debug=False).to(device)
    state_dict=torch.load('model.pth',map_location=torch.device(device))
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()
        for idx, sample_batch in enumerate(test_dataloader, 0):
            outputs = model(sample_batch['audio'].unsqueeze(1).to(device), sample_batch['video'].to(device), sample_batch['touch'].to(device), sample_batch['pose'].unsqueeze(1).to(device))
            print("Index",idx,'----\n',
            "    Index",idx,'----',"Prediction",outputs.to(device),'\n',
            "    Index",idx,'----',"Ground Truth",sample_batch['label'].to(device),'\n',
            "    Index",idx,'----',"Loss",outputs.to(device)-sample_batch['label'].to(device),'\n')