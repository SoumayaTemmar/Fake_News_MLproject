# Fake_News_MLproject 
in thi project we will train a ML model to detect whether a news article is real or fake using textual features

dataset used: [Fake News Classification - Kaggle]:  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Dataset contains four columns:
- **Serial number** (starting from 0) 
- **Title** (about the text news heading) 
- **Text** (about the news content)
- **Label** (0 = fake and 1 = real) as specified in Kaggle documentation.

# 1-Features / Key Components

**Data Ingestion:** Cleaning, deduplication, language detection, sentiment scoring.
**Data Transformation:** TF-IDF vectorization, scaling numeric features.
**Model Training:** Logistic Regression
**Predict Pipeline:** Takes new articles, cleans them,calculate numeric diagnostics and predicts label.
**Evaluation:** Accuracy, precision, recall, F1-score.

# 2-Project structure
Fake_News_MLProject/
│
├── artifacts/          # Preprocessed data, trained model, preprocessor
├── src/                # Source code
│   ├── components/     # Data ingestion, transformation, model trainer, pipelines
│   ├── utils.py        # Helper functions
│   ├── exception.py    # Custom exceptions
│   └── logger.py       # Logging setup
├── notebooks/          # Exploratory data analysis / experimentation
├── requirements.txt    # Project dependencies
├── setup.py            # for package building
└── README.md           # This file


# 3-Installation & Setup
# Clone the repo
git clone <repo-url>
cd Fake_News_MLProject

# Create a virtual environment
conda create -p venv python=3.8 -y
conda activate venv/

# Install dependencies
pip install -r requirements.txt

# 4-Train the model
python src/pipelines/train_pipeline.py

# 5- run the app
python app.py 

- This launches a Flask app on http://localhost:5000 where you can paste news articles and get real-time predictions.
