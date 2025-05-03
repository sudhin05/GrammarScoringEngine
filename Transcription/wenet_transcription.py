import wenet
import os
import logging 

def transcribe_audio(audio_dir,mode='test'):
    logging.basicConfig(level=logging.INFO,filename=f"{mode}_files.log",filemode="w")
    model = wenet.load_model('english')
    ctr = 1
    for aud in sorted(os.listdir(audio_dir)):
        if ctr % 44 == 0:
            logging.debug("Info: 44 files processed")
            break
        apath = os.path.join(audio_dir,aud) 
        result = model.transcribe(apath)
        logging.info(f"{aud} Processed, Transcription is: {result}")


if __name__ == "__main__":
    transcribe_audio("Dataset/audios/test")