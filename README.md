---
title: Football Event Intelligence
emoji: ⚽
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚽ Explainable Football Event Classification

This project develops an explainable deep-learning system for classifying football images into seven event categories. It compares a custom convolutional neural network with two transfer-learning models and includes model evaluation, explainability analysis, and deployment through a web application.

## Project objective

The main objective is to determine how accurately football events can be classified from still images and whether pretrained models perform better than a CNN trained from scratch. The project also investigates whether the models focus on meaningful football regions instead of irrelevant background details.

## Event classes

- Corner Kick
- Free Kick
- Penalty Kick
- Red Card
- Yellow Card
- Substitute
- Tackle

## Models evaluated

1. **Custom CNN** — baseline model trained from scratch.
2. **EfficientNetV2** — pretrained model adapted using transfer learning and fine-tuning.
3. **ConvNeXtTiny** — pretrained model adapted using transfer learning and fine-tuning.

## Dataset preparation

The original dataset contained approximately 6,168 usable images. Duplicate and conflicting images were removed to reduce data leakage and improve the reliability of the evaluation. The cleaned dataset contained approximately 4,667 unique images and was divided into training, validation, and test sets.

| Split | Images |
|---|---:|
| Training | 3,266 |
| Validation | 700 |
| Test | 701 |

The dataset source is available in the [`datasets`](datasets/) folder.

## Main results

ConvNeXtTiny achieved the strongest overall performance.

| Model | Test Accuracy | Macro F1 | Top-2 Accuracy |
|---|---:|---:|---:|
| Custom CNN | 61.6% | 60.0% | 84.0% |
| EfficientNetV2 | 83.0% | 82.6% | 92.4% |
| ConvNeXtTiny | 84.6% | 84.3% | 94.4% |

The results show that transfer learning clearly outperformed the custom CNN trained from scratch.

## Explainable AI

- **Grad-CAM** highlights the image regions that influenced the CNN prediction.
- **SHAP** shows which regions contributed positively or negatively to a prediction.

These methods were used to verify whether the model focused on meaningful football details such as players, referees, the ball, cards, and action areas.

## Repository structure

```text
MachineLearningFinalProjectKEAB/
├── notebooks/                 # Final training and evaluation notebook
├── figures/                   # Report-ready figures and model results
├── datasets/                  # Dataset source information
├── app.py                     # FastAPI application and frontend routes
├── best_football_classifier.keras
├── class_mapping.json
├── deployment_config.json
├── Dockerfile
├── requirements.txt
└── README.md
```

## Web application

The best-performing ConvNeXtTiny model was deployed in a Docker-based Hugging Face Space. Users can upload a football image and receive the predicted event class, confidence scores, and ranked class probabilities.

### Application routes

- `/` — web interface
- `/health` — model status
- `/predict` — prediction endpoint
- `/docs` — API documentation

## Running the project locally

### 1. Clone the repository

```bash
git clone https://github.com/Kalvarez09/MachineLearningFinalProjectKEAB.git
cd MachineLearningFinalProjectKEAB
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860` in a browser.

## Technologies used

- Python
- TensorFlow / Keras
- scikit-learn
- NumPy
- pandas
- Matplotlib
- SHAP
- FastAPI
- Docker
- Hugging Face Spaces

## Author

**Karim Eduardo Alvarez Becerra**  
University of Europe for Applied Sciences, Potsdam, Germany

## License

This project is released under the MIT License.
