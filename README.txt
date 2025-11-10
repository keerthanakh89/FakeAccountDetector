FakeAccountDetector - Mini Project
=================================

This project was prepared for Keerthana's mini-project on Fake Social Media Account Detection.

How to run (in VS Code):
1. Open the folder 'FakeAccountDetector' in VS Code.
2. (Optional) Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate   (Windows) or source venv/bin/activate (Linux/Mac)
3. Install dependencies:
   pip install -r requirements.txt
4. Train models:
   python train_model.py
   This will read dataset/train.csv, train DecisionTree, RandomForest, and SVM, then save the best model to model/fake_model.pkl
   It will also save a model comparison plot to model/model_comparison.png
5. Run the web app:
   python app.py
6. Open http://127.0.0.1:5000 in your browser.

Notes:
- The automatic feature selection in train_model.py tries common column names. If your train.csv uses different column names, open train_model.py and adjust 'possible_feature_cols' and 'label_cols'.
- Presentations included: project_presentation_1.pptx and project_presentation_2.pptx