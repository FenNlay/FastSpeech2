# FastSpeech 2 - EmoFS2
This is the FastSPeech 2 model used to finish a thesis. It is based on the implementation of [ming024](https://ming024.github.io/FastSpeech2/).

# Use
In line with the forked GitHub, this FastSpeech 2 model works on the LibriTTS workflow. Instead of multiple speakers, multiple emotions are used to train he model on.

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference


After training, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --speaker_id SPEAKER_ID --restore_step 100000 --mode single -p config/EmoFS2/preprocess.yaml -m config/EmoFS2/model.yaml -t config/EmoFS2/train.yaml
```

Different speaker_id's are:
0 -> neutral
1 -> happy
2 -> angry
3 -> sad
4 -> surprise

The generated utterances will be put in ``output/result/``.

# Training

## Datasets

The used dataset is

- [ESD](https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view): multiple speakers, two languages, 5 emotional states; each speaker dataset consists of the same 350 short audio clips per emotion.

Only speaker 14 is used.

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/EmoFS2/preprocess.yaml
```

After that, run the preprocessing script by
```
python3 preprocess.py config/EmoFS2/preprocess.yaml
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/EmoFS2/preprocess.yaml
```

## Training

Train your model with
```
python3 train.py -p config/EmoFS2/preprocess.yaml -m config/EmoFS2/model.yaml -t config/EmoFS2/train.yaml
```
