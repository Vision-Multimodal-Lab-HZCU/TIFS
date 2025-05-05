## Environment

We conduct experiments with Python 3.8.13 and Pytorch 1.13.0.

To setup the environment, please simply run

```
pip install -r requirements.txt
```

## Dataset & Extract
For [Touch and Go](Touch and Go (touch-and-go.github.io)) dataset, please download and unzip the 'dataset' folder containing video files to 'TIFS/raw_data/touch_ond_go'. Next, for videos extract, please run

```
cd raw_data/touch_and_go
python extract_frame.py
```

For [VisGel](https://github.com/YunzhuLi/VisGel) dataset, please download and unzip the 'data_seen' to 'TIFS/raw_data/visgel'.

## Visual Imagination Synthesis
After downloading and extracting the dataset, the next step is to perform visual imagination synthesis. Please run

```
cd IVS
python IVS.py
python IVScopy.py
```

## Feature Extraction
Using pre-trained model VideoMAE for feature extraction, please run

```
cd genfea
python gen_visual.py
python gen_touch.py
```

## Training & Evaluation
Please run

```
sh run_incremental_ours.sh
```
