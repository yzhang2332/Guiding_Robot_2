from datasets import DogDataset
import torchvision
from torch.utils.data import DataLoader


if __name__=="__main__":
    '''Test the dataset with the sample data.'''
    test_dataset = DogDataset(annotations_file='modal_fusion/datasets/test_data/processed_data/annotations.csv', 
                              audio_dir="modal_fusion/datasets/test_data/processed_data/audio", 
                              video_dir="modal_fusion/datasets/test_data/processed_data/video", 
                              touch_dir="modal_fusion/datasets/test_data/processed_data/touch", 
                              pose_dir="modal_fusion/datasets/test_data/processed_data/pose"
                              )
    
    for i, sample in enumerate(test_dataset):
        print(i, sample['audio'].shape, sample['video'].size(), sample['touch'].size(), sample['pose'].size(), sample['label'].size())
        # torchvision.transforms.ToPILImage()(sample['touch']).show()
        # torchvision.transforms.ToPILImage()(sample['video'][0][0]).show()
        print(sample['touch'])
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(test_dataloader):
        print(i_batch, sample_batched['audio'].size(), sample_batched['video'].size(), sample_batched['touch'].size(), sample_batched['pose'].size(), sample_batched['label'].size())
