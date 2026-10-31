# AI Helmet: Camera-Independent Two-Wheeler Crash-Risk Prediction

Course MDI4001, VIT. Team: Jashwanth R (24MID0274), Anusri J (24MID0002). Faculty in charge: Rohith R.

AI Helmet estimates the chance of a crash in the next few seconds from ordinary video. One model is trained to work on any camera view: the dashcam of another vehicle, a cyclist's point of view, or a helmet, webcam or phone camera. An R Shiny dashboard shows the live camera, saved rides, crash maps and model results.

## How it works
1. **Detect and track:** YOLO11n + ByteTrack find and follow cars, bikes, people and trucks. Farneback optical flow measures camera and scene motion.
2. **Camera-independent features:** 25 numbers per one-second window (looming, time-to-contact, sideways miss distance, pair overlap and closing rate, scene flow). They describe relative change, so they do not depend on lens, resolution, frame rate or mounting.
3. **Model:** five classifiers compared (logistic regression, random forest, HistGradientBoosting, XGBoost, soft-voting ensemble). The voting ensemble is calibrated and used with an alert threshold of 0.49 and a two-window alert rule.
4. **Evaluation:** 5-fold cross-validation grouped by clip, source-balanced weights, and leave-one-source-out testing.

## Results (measured on clips the model did not train on)
| | ROC AUC |
|---|---|
| All sources (voting ensemble) | 0.78 |
| CCD / Cyclist / Kaggle, clips held out | 0.75 / 0.80 / 0.86 |
| Camera type never seen in training (CCD / Cyclist / Kaggle) | 0.67 / 0.69 / 0.61 |

With two consecutive windows required, 61.6% of crash clips are warned about, a median of 1.4 s ahead, with about 2.1 false alerts per minute. This is a research prototype. Generalization to an unseen camera type is the main weakness, Kaggle crash times are assumed to be near the end of each clip, and no helmet-camera footage with known crash times was available yet.

## Folder structure
```
src/                 Python code: feature extraction, training, prediction
r_analysis/          R plots and the Shiny dashboard (live_dashboard/app.R)
dataset/             labels and result CSVs (videos are not included)
models/              trained model (universal_crash_model.joblib)
requirements.txt
```

## Data (not stored in this repository)
- CCD Crash-1500 and Normal clips: https://github.com/Cogito2012/CarCrashDataset
- Kaggle accident / driving clips: [add link]
- Cyclist clips: [add link]
- NCDOT bicycle and pedestrian crash data (North Carolina, 2007-2018): [add link]
- Full dataset copy: [add Google Drive link if you share one]

Put the videos in `dataset/ccd_videos`, `dataset/kaggle_videos` and `dataset/clips_output`, with the labels files in `dataset/`.

## Run
```
pip install -r requirements.txt

# 1. Extract features once per source (add --target-fps 15 to speed up)
python src/universal_collector.py --videos dataset/ccd_videos --labels dataset/labels_ccd.csv --output dataset/training_universal_ccd.csv --source-name ccd --viewpoint other_vehicle_dashcam
python src/universal_collector.py --videos dataset/kaggle_videos --labels dataset/labels_kaggle.csv --output dataset/training_universal_kaggle.csv --source-name kaggle --viewpoint other_vehicle_dashcam
python src/universal_collector.py --videos dataset/clips_output --labels dataset/labels_cyclist.csv --output dataset/training_universal_cyclist.csv --source-name cyclist --viewpoint cyclist_pov

# 2. Fix CCD crash times from the official annotation file (Crash-1500.txt)
python src/relabel_ccd.py --annotation dataset/Crash-1500.txt --videos dataset/ccd_videos

# 3. Combine and train
python src/combine_universal_data.py
python src/train_universal_model.py

# 4. Run on any camera
python src/universal_predict.py --source 0 --live-log
python src/universal_predict.py --source "http://<phone-ip>:8080/video" --live-log
python src/universal_predict.py --source "video.mp4" --save-ride demo1
```
Dashboard: open `r_analysis/live_dashboard/app.R` in RStudio, run `source("r_analysis/install_packages.R")` once, then click Run App and choose Live camera feed.
