from moviepy.editor import *
import os
import pandas as pd


# def slideVideo(from_dir=None, to_dir=None, period=6, step=1):
#     '''Slide on the video with a window to generate subclips.
    
#     Arguments:
#         from_dir (string): Directory of the original videos.
#         to_dir (string): Directory that new short videos will go to.
#         period (float or int): Time interval that the subclip spans in seconds (15.35).
#         step (float or int): Time shift between two adjacent subclips.
#     '''
#     files = os.listdir(from_dir)

#     for file in files:
#         # read video
#         video = VideoFileClip(os.path.join(from_dir, file))
#         # read the video name without extension
#         name = file.rpartition('.')[0]
#         # cut out a period-duration video every step
#         t_start = 0
#         t_end = period
#         if t_end > video.duration:
#             video.write_videofile(os.path.join(to_dir, name+'_'+str(t_start)+'_'+str(video.duration)+'.mp4'))
#             continue
#         while t_end <= video.duration:
#             video.subclip(t_start, t_end).write_videofile(os.path.join(to_dir, name+'_'+str(t_start)+'_'+str(t_end)+'.mp4'))
#             t_start += step
#             t_end += step
#             if t_end > video.duration:
#                 video.subclip(t_start, video.duration).write_videofile(os.path.join(to_dir, name+'_'+str(t_start)+'_'+str(video.duration)+'.mp4'))


# def sepAudioVideo(from_dir=None, audio_dir=None, video_dir=None, audio_ext='.wav'): 
#     '''Separate the audio and video from the original video.
    
#     Arguments:
#         from_dir (string): Directory of the original videos with audios.
#         audio_dir (string): Directory that separated audios will go to.
#         video_dir (string): Directory that separated videos will go to.
#     '''
#     files = os.listdir(from_dir)

#     for file in files:
#         # read video
#         video = VideoFileClip(os.path.join(from_dir, file))
#         # read the video name without extension
#         name = file.rpartition('.')[0]
#         # save audio using the name
#         video.audio.write_audiofile(os.path.join(audio_dir, name+audio_ext))
#         # save silent video
#         video.without_audio().write_videofile(os.path.join(video_dir, file))


# def fillPose(from_dir=None, to_dir=None):
#     '''Fill the missing data in the pose csv files with estimated data.
    
#     Arguments:
#         from_dir (string): Directory of the original pose csv files.
#         to_dir (string): Directory of the filled pose csv files.
#     '''
#     files = os.listdir(from_dir)
#     pure_data_path = os.path.join(from_dir, 'test_pose_pure.csv')
#     for file in files:
#         # read useful data
#         with open(os.path.join(from_dir, file)) as f:
#             data = f.readlines()[6:]
#         f.close()
#         # write useful data
#         with open(pure_data_path, 'w') as f:
#             f.writelines(data)
#         f.close()
#         # read dataframe of the useful data and interpolate and write new file
#         df = pd.read_csv(pure_data_path, index_col='Frame')
#         new_df = df.interpolate(method='cubicspline')
#         new_df.to_csv(os.path.join(to_dir, file))
#     # delete the pure data file
#     os.remove(pure_data_path)


# def slidePose(from_dir=None, to_dir=None, period=6, step=1, fps=120):
#     '''Slide on the pose csv files to generate subposes.
    
#     Arguments:
#         from_dir (string): Directory of the original pose csv files.
#         to_dir (string): Directory that new short pose csv files will go to.
#         period (int): Time interval that the subpose spans in seconds.
#         step (int): Time shift between two adjacent subposes in seconds.
#         fps (int): The number of frames per second when recording the poses.
#     '''
#     files = os.listdir(from_dir)

#     for file in files:
#         # read the pose csv file
#         df = pd.read_csv(os.path.join(from_dir, file), index_col='Frame')
#         # read the file name without extension
#         name = file.rpartition('.')[0]
#         # cut out period-duration frames of poses every step
#         f_start = 1
#         f_end = period*fps
#         if f_end > df.shape[0]:
#             df.to_csv(os.path.join(to_dir, name+'_'+str(f_start)+'_'+str(df.shape[0])+'.csv'))
#             continue
#         while f_end <= df.shape[0]:
#             df.loc[f_start:f_end].to_csv(os.path.join(to_dir, name+'_'+str(f_start)+'_'+str(f_end)+'.csv'))
#             f_start += step*fps
#             f_end += step*fps
#             if f_end > df.shape[0]:
#                 df.loc[f_start:df.shape[0]].to_csv(os.path.join(to_dir, name+'_'+str(f_start)+'_'+str(df.shape[0])+'.csv'))


def preprocess(annotations_file=None, original_video_dir=None, original_pose_dir=None, original_touch_dir=None, 
               audio_dir=None, video_dir=None, touch_dir=None, pose_dir=None, 
               period=6, step=1, fps=120, 
               normalize=True):
    '''Generate subaudios, subvideos, subtouches and subposes based on the annotations_file with original data.
    
    Arguments:
        annotations_file (string): Directory of the original annotation file.
        original_video_dir (string): Directory of the original video files.
        original_pose_dir (string): Directory of the original pose files.
        original_touch_dir (string): Directory of the original touch files.
        audio_dir (string): Directory that new short audio files will go to.
        video_dir (string): Directory that new short video files will go to.
        touch_dir (string): Directory that new short touch files will go to.
        pose_dir (string): Directory that new short pose files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        fps (int): The number of frames per second when recording the poses.
        normalize (boolean): The boolean setting if the pose values are normalized.
    '''
    # read original annotations
    init_annotations = pd.read_csv(annotations_file)

    # write new annotation-file head
    with open('modal_fusion/datasets/test_data/processed_data/annotations.csv', 'w') as f:
        f.write('audio,video,touch,pose,label\n')
    f.close()

    # traverse through lines of data instances
    for i in range(init_annotations.shape[0]):

        # a line (an instance)
        data_instance = init_annotations.iloc[i]
        # find indicated files
        video_path = os.path.join(original_video_dir, data_instance['video'])
        pose_path = os.path.join(original_pose_dir, data_instance['pose'])
        # touch_path = os.path.join(original_touch_dir, data_instance['touch'])
        # preprocess the original video, touch, pose file; generate clips and labels
        audio_clip_names, video_clip_names = preprocessVideo(video_path, audio_dir, video_dir, period=period, step=step)
        pose_clip_names, labels = preprocessPose(pose_path, pose_dir, period=period, step=step, fps=fps, normalize=normalize)
        # labels就是分割的pose的数据内容

        # align the clip_name lists for new annotations
        if len(audio_clip_names) > len(pose_clip_names):
            diff = len(audio_clip_names) - len(pose_clip_names)
            del audio_clip_names[-diff:]
            del video_clip_names[-diff:]
        elif len(audio_clip_names) < len(pose_clip_names):
            diff = len(pose_clip_names) - len(audio_clip_names)
            del pose_clip_names[-diff:]
            del labels[-diff:]
        # write the new annotation file
        for audio, video, pose, label in zip(audio_clip_names, video_clip_names, pose_clip_names, labels):
            line = audio + ',' + video + ',' + data_instance['touch'] + ',' + pose + ',' + '_'.join([str(item) for item in label.values.tolist()]) + '\n'
            with open('modal_fusion/datasets/test_data/processed_data/annotations.csv', 'a') as f:
                f.write(line)
            f.close()


def preprocessVideo(video_path=None, to_audio_dir=None, to_video_dir=None, period=6, step=1, audio_ext='.wav', video_ext='.mp4'):
    '''Generate subaudios, subvideos from original video.
    
    Arguments:
        video_path (string): Directory of the original video.
        to_audio_dir (string): Directory that new short audio files will go to.
        to_video_dir (string): Directory that new short video files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        audio_ext (string): The audio file extension.
        video_ext (string): The video file extension.
    '''
    # read video
    video = VideoFileClip(video_path)
    # read the video name without extension
    name = os.path.basename(video_path).rpartition('.')[0]
    # use a window to slide on the video and separate the audio and video parts
    t_start = 0
    t_end = period
    if t_end > video.duration:
        print(f'{video_path} is too short to slide on!')
        return
    audio_clip_names, video_clip_names = [], []
    while t_end <= video.duration:
        video_clip = video.subclip(t_start, t_end).set_fps(1)
        name_clip = name+'_'+str(t_start)+'_'+str(t_end)
        video_clip.audio.write_audiofile(os.path.join(to_audio_dir, name_clip+audio_ext))
        video_clip.without_audio().write_videofile(os.path.join(to_video_dir, name_clip+video_ext))
        audio_clip_names.append(name_clip+audio_ext)
        video_clip_names.append(name_clip+video_ext)
        t_start += step
        t_end += step
        
    return audio_clip_names, video_clip_names


def preprocessPose(pose_path=None, to_pose_dir=None, period=6, step=1, fps=120, normalize=True):
    '''Generate subposes from original pose csv file.
    
    Arguments:
        pose_path (string): Directory of the original pose file.
        to_pose_dir (string): Directory that new short pose files will go to.
        period (int): Time interval that the new files span in seconds.
        step (int): Time shift between two adjacent clips in seconds.
        fps (int): The number of frames per second when recording the poses.
    '''
    # fill missing values of the original pose file and read it
    pure_data_path = os.path.join(to_pose_dir, 'pose_pure.csv')
    with open(pose_path) as f:
        data = f.readlines()[6:]
    f.close()
    with open(pure_data_path, 'w') as f:
        f.writelines(data)
    f.close()
    df = pd.read_csv(pure_data_path, index_col='Frame')
    df = df.interpolate(method='cubicspline')
    # normalize the pose csv file values
    if normalize:
        max_values = []
        for column in df.columns:
            maximum = df[column].abs().max()
            if maximum != 0:
                df[column] = df[column] / maximum
            max_values.append(maximum)
        with open('modal_fusion/preprocess_max_values_normalize.txt', 'w') as fp:
            fp.write("\n".join(str(item) for item in max_values))
    os.remove(pure_data_path)
    # read the pose file name without extension
    name = os.path.basename(pose_path).rpartition('.')[0]
    # use a window to slide on the pose file and separate out subposes
    t_start = 0
    t_end = period
    f_start = 1
    f_end = period*fps
    if f_end > df.shape[0]:
        print(f'{pose_path} is too short to slide on!')
        return
    pose_clip_names, labels = [], []
    while f_end <= df.shape[0]:
        pose_clip = df.loc[f_start:f_end]
        name_clip = name+'_'+str(t_start)+'_'+str(t_end)+'.csv'
        pose_clip.to_csv(os.path.join(to_pose_dir, name_clip))
        pose_clip_names.append(name_clip)
        label_index = f_end+step*fps
        if label_index > df.shape[0]:
            label = df.loc[df.shape[0]]
        else:
            label = df.loc[label_index]
        labels.append(label)
        t_start += step
        t_end += step
        f_start += step*fps
        f_end += step*fps

    return pose_clip_names, labels



if __name__=="__main__":
    
    # slideVideo(from_dir='modal_fusion/datasets/test_data/a+v_raw', to_dir='modal_fusion/datasets/test_data/a+v_clips')
    # sepAudioVideo(from_dir='modal_fusion/datasets/test_data/a+v_clips', audio_dir='modal_fusion/datasets/test_data/processed_data/audio', video_dir='modal_fusion/datasets/test_data/processed_data/video')
    # fillMissingVal(from_dir='modal_fusion/datasets/test_data/pose_raw', to_dir='modal_fusion/datasets/test_data/pose_filled')
    # slidePose(from_dir='modal_fusion/datasets/test_data/pose_filled', to_dir='modal_fusion/datasets/test_data/pose_clips')
    
    preprocess(annotations_file='modal_fusion/datasets/test_data/original_annotations.csv', 
               original_video_dir='modal_fusion/datasets/test_data/raw_video',
               original_pose_dir='modal_fusion/datasets/test_data/raw_pose',
               audio_dir='modal_fusion/datasets/test_data/processed_data/audio', 
               video_dir='modal_fusion/datasets/test_data/processed_data/video', 
               pose_dir='modal_fusion/datasets/test_data/processed_data/pose')