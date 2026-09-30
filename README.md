# RSA: Rényi Sine Adaptation for Robust Multi-Modal Test-Time Adaptation

## Overview

Package Requirements: torch, torchaudio, timm, scikit-learn, numpy

# Data Preparation

We provide two benchmarks upon the VGGSound and Kinetics datasets for the challenge following the dataset construction as [READ (ICLR 24)](https://github.com/XLearning-SCU/2024-ICLR-READ):

You can download the benchmarks via [Google Cloud](https://drive.google.com/drive/folders/1SWkNwTqI08xbNJgz-YU2TwWHPn5Q4z5b?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1Xo3IxQyd_fkzMVofDWKYVw?pwd=fnha) and use them according to the following steps. Notably, to make it easier, it is recommended to only modify the default `root_path`.

**Step 1. Introduce corruptions for either video or audio modality**

Modify the `path` and `save-path`, and specify the `corruption` and `severity` to introduce the corruptions with different severity levels.

### Video-corruption:

```bash
python ./make_corruptions/make_c_video.py --corruption 'gaussian_noise' --severity 5 --data-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --save-path 'data_path/Kinetics50/image_mulframe_val256_k-C'
```

### Audio-corruption:

```bash
python ./make_corruptions/make_c_audio.py --corruption 'gaussian_noise' --severity 5 --data_path 'data_path/Kinetics50/audio_val256_k=50' --save-path 'data_path/Kinetics50/audio_val256_k=50-C' --weather_path 'data_path/weather_audios/'
```

**Step 2. Create JSON files for benchmarks**

JSON files are needed to reproduce our code. To make it easier, we provide our JSON files as references and you can customize your own JSON files by using the following scripts.

### JSON file for non-corrupted data (**Mandatory**):

```bash
python ./data_process/create_clean_json.py --refer-path 'code_path/json_csv_files/ks50_test_refer.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --save-path 'code_path/json_csv_files/ks50'
```

### JSON file for video-corrupted data:

```bash
python ./data_process/create_video_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-c-path 'data_path/Kinetics50/image_mulframe_val256_k=50-C' --audio-path 'data_path/Kinetics50/audio_val256_k=50' --corruption 'gaussian_noise'
```

### JSON file for audio-corrupted data:

```bash
python ./data_process/create_audio_c_json.py --clean-path 'code_path/json_csv_files/ks50/clean/severity_0.json' --video-path 'data_path/Kinetics50/image_mulframe_val256_k=50' --audio-c-path 'data_path/Kinetics50/audio_val256_k=50-C' --corruption 'gaussian_noise'
```

### Running the code

You can run the code using the command:

```bash
python main.py --dataset 'ks50' --json-root [json-root] --label-csv [label-csv] --pretrain_path [pretrain_path] --tta-method 'rsa' --severity 55 --corruption-modality 'both'
```

## Acknowledgements

Our project referenced the code of the following repositories. We sincerely thank them for offering a useful public code base.

[READ (ICLR 24)](https://github.com/XLearning-SCU/2024-ICLR-READ)
