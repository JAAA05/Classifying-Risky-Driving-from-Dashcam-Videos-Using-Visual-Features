# Classifying Risky Driving from Dashcam Videos Using Visual Features

This README explains the basic steps to set up and run the project.

**Note:** This setup was tested on Ubuntu 24.04.

## Requirements

Install Pixi:

https://pixi.prefix.dev/latest/installation/

## 1. Clone the Repository

```bash
git clone https://github.com/JAAA05/Classifying-Risky-Driving-from-Dashcam-Videos-Using-Visual-Features.git
```

Go into the project folder:

```bash
cd Classifying-Risky-Driving-from-Dashcam-Videos-Using-Visual-Features/
```

## 2. Create the Pixi Environment

```bash
pixi install
```

## 3. Download and Prepare the Dataset

Start the Pixi shell:

```bash
pixi shell
```

Download the dataset:

```bash
gdown 1iP91Kg2dgJFTbZ6xhF506gK2w1EftY89
```

Unzip the dataset:

```bash
unzip 2026-04-20.zip
```

Create the videos folder:

```bash
mkdir videos
```

Move the dataset into the videos folder:

```bash
mv 2026-04-20 videos/
```

## 4. Feature Extraction

Run the batch processing script:

```bash
python src/pipeline/batch_process.py
```

## 5. Alternative Dataset Generation

To create the split dataset, run:

```bash
python src/utils/split_dataset.py
```

To create the split + random dataset, run:

```bash
python src/utils/split_dataset_random.py
```

## 6. Model Training and Evaluation

To train and evaluate the model, run:

```bash
python src/models/train_classifier.py
```

This same training script is used after preparing the dataset version that you want to test, such as the original dataset, split dataset, or split + random dataset.

## 7. Ablation Study

To run the ablation study, use:

```bash
python src/models/ablation_study.py
```
