import os
import re
import librosa
import pandas as pd
import matplotlib.pyplot as plt

def extract_no(filename):
    match = re.search(r'audio_(\d+)\.wav', filename)
    return int(match.group(1)) if match else -1

def get_audio_info(audio_dir):
    data = []
    files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    files = sorted(files, key=extract_no)

    for file in files:
        path = os.path.join(audio_dir, file)
        try:
            y, sr = librosa.load(path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            data.append({'filename': file, 'sample_rate': sr, 'duration_sec': duration})
        except Exception as e:
            print(f"Error hai in {file}: {e}")

    return pd.DataFrame(data)

def plot_aud_distri(df,col='duration_sec'):
    plt.figure(figsize=(10, 6))
    plt.hist(df[col], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distri of aud file {col}')
    plt.xlabel(f'{col})')
    plt.ylabel('Number of Files')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def flag_audio_info(df,csv_path):
    df_long = df[df['duration_sec'] > 70]
    df_long_sorted = df_long.sort_values(by='duration_sec', ascending=True)

    df_long_sorted.to_csv(csv_path, index=False)
    print(f"Saved {len(df_long_sorted)} long files to {csv_path}")

def mod_flag_audio_info(df_all,df_flagged,csv_path):
    df_merged = pd.merge(df_flagged, df_all[['filename', 'label']], on='filename', how='left')
    df_merged.to_csv(csv_path, index=False)
    print(f"Merged CSV saved as: {csv_path}")


if __name__ == "__main__":
    audio_dir = 'Dataset/audios/train' 
    labels_csv = 'Dataset/train.csv'
    output_csv = 'logs/csvs/audio_prop.csv'
    output2_csv = 'logs/csvs/flag_audio_prop.csv'
    output3_csv = 'logs/csvs/modflags_audio_prop.csv'


    # df = get_audio_info(audio_dir)
    # df.to_csv(output_csv, index=False)
    # print(f"Metadata saved to {output_csv}")


    # df = pd.read_csv(output_csv)
    # plot_aud_distri(df)


    # df = pd.read_csv(output_csv)
    # flag_audio_info(df,output2_csv)

    # df_all = pd.read_csv(labels_csv)
    # df_flagged = pd.read_csv(output2_csv)
    # mod_flag_audio_info(df_all,df_flagged,output3_csv)

    df_mod = pd.read_csv(output3_csv)
    plot_aud_distri(df_mod,col='label')




