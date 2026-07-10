# Plant Disease Classification

This project deploys a PyTorch `EfficientNet-B0` multi-label classifier with Streamlit Community Cloud.

## Root files required for deployment

- `app.py`
- `requirements.txt`
- `README.md`
- `plant_model_demo.pth`

## Model details

- Architecture: `EfficientNet-B0`
- Output classes:
  - `complex`
  - `frog_eye_leaf_spot`
  - `healthy`
  - `powdery_mildew`
  - `rust`
  - `scab`
- Output activation: `sigmoid`

## Streamlit app behavior

- Upload `jpg`, `jpeg`, or `png` images
- Resize to `224x224`
- Normalize with Albumentations
- Convert to tensor with `ToTensorV2`
- Predict multi-label outputs from `plant_model_demo.pth`
- Control displayed labels with a threshold slider

## Deploy to Streamlit Community Cloud

Push this repository to GitHub, then create a new Streamlit app with:

- Repository: `YOUR_USERNAME/plant-disease-classification`
- Branch: `main`
- Main file path: `app.py`

The app does not depend on Kaggle dataset paths. It only loads the saved model file `plant_model_demo.pth`.
