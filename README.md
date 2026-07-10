# Plant Disease Classification

This project deploys a PyTorch `EfficientNet-B0` multi-label classifier with Streamlit Community Cloud.

Live app:

- https://plant-disease-classification-ml.streamlit.app/

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

## Test Images

Sample test images are included in [test-images](./test-images) so users can download them and try the app immediately.

- [complex_800cbf0ff87721f8.jpg](./test-images/complex_800cbf0ff87721f8.jpg)
- [healthy_801d6dcd96e48ebc.jpg](./test-images/healthy_801d6dcd96e48ebc.jpg)
- [rust_800f85dc5f407aef.jpg](./test-images/rust_800f85dc5f407aef.jpg)

To test the hosted app:

1. Open the Streamlit app link above.
2. Download any image from the `test-images/` folder.
3. Upload it in the app.
4. Adjust the prediction threshold if needed.


## Streamlit app behavior

- Upload `jpg`, `jpeg`, or `png` images
- Resize to `224x224`
- Normalize with Albumentations
- Convert to tensor with `ToTensorV2`
- Predict multi-label outputs from `plant_model_demo.pth`
- Control displayed labels with a threshold slider
- Show class probabilities for all 6 classes

## What Prediction Threshold Means

The model outputs a probability score for each class.

- If a class score is greater than or equal to the selected threshold, that label is shown as predicted.
- Lower threshold: more labels may appear.
- Higher threshold: fewer labels appear, but they must have stronger confidence.

Example:

- At threshold `0.50`, any class with probability `0.50` or higher is shown.
- If `rust = 0.82`, it will be included.
- If `scab = 0.34`, it will not be included.
