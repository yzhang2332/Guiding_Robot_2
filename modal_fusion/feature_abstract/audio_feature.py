import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioFeatureExtract:
    def __init__(self,audio_addr,n_mfcc=13,frame_length = 0,hop_length=256,win_length=1024,debug=False,mono=True) -> None:
        self.mono=mono
        self.audio, self.sr = librosa.load(str(audio_addr), sr=None, mono=self.mono)
        # self.audio = audio
        # self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.debug = debug

    def extract_mfcc(self): 
        if self.debug:
            print("shape:", self.audio.shape, "Sample Rate: ",self.sr,"Duration: ",librosa.get_duration(y=self.audio))

        mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr)
        mfcc_v = librosa.feature.delta(mfcc, width = 5)
        mfcc_acc = librosa.feature.delta(mfcc_v, width = 5)
        mfcc_final = np.transpose(np.concatenate((mfcc, mfcc_v, mfcc_acc)))
        
        if self.debug:
            fig, ax = plt.subplots(nrows=2, sharex=True)
            img = librosa.display.specshow(librosa.power_to_db(
                librosa.feature.melspectrogram(y=self.audio, sr=self.sr), ref=np.max),
                x_axis='time', y_axis='mel', fmax=8000,ax=ax[0])
            fig.colorbar(img, ax=[ax[0]])
            ax[0].set(title='Mel spectrogram')
            ax[0].label_outer()
            img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
            fig.colorbar(img, ax=[ax[1]])
            ax[1].set(title='MFCC')
            plt.show()

        return mfcc_final
    


if __name__ == '__main__':

    audiotest = AudioFeatureExtract(audio_addr="wav.wav",debug=True)
    mfccFeatures = audiotest.extract_mfcc()
    print("MFCC_features: ",mfccFeatures, "\nshape:",mfccFeatures.shape)