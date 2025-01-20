
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import create_data
import test_model_new
import numpy as np
import draw_confidence_ellipse
from sklearn.preprocessing import label_binarize
from itertools import cycle
import means_covs
from sklearn.metrics import roc_curve, auc
import os

# הגדרת פרמטרים
num_samples = 10000
num_classes=2
train_ratio = 0.8
train_ratio = 0.8
spread = 10
overlap_percentage = 0.001 # אחוז חפיפה
sigma = 1.0


# יצירת תוחלות ומטריצות שונות בהתבסס על אחוז חפיפה
means, covs = means_covs.generate_means_covs(
    num_classes, overlap_percentage=overlap_percentage, sigma=sigma, random_state=42
)

#means = [[-3, -3], [3, -3], [3, 3]]  # תוחלת לכל ענן
#covs = [
    #[[1.0, 0.3], [0.3, 1.0]],  # מחלקה 0
    #[[1.5, 0.5], [0.5, 1.5]],  # מחלקה 1
    #[[0.8, 0.2], [0.2, 0.8]]   # מחלקה 2
#]


X,y = create_data.create_data_fn(num_samples,num_classes,means,covs)

# חלוקת הנתונים
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

# יצירת המודל
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)


# הגדרת הנתיב שבו הסקריפט רץ
current_dir = os.path.dirname(os.path.abspath(__file__))

# קביעת הנתיב לשמירה יחסית לתיקייה שבה נמצא הסקריפט
decision_boundary_path = os.path.join(current_dir, 'decision_boundaries.png')
roc_curve_path = os.path.join(current_dir, 'roc_curve.png')

X_test, y_test, y_pred, y_pred_proba = test_model_new.test_model_fn(
    X, y, train_ratio, model, classes=list(range(num_classes)),
    decision_boundary_path=decision_boundary_path,
    roc_curve_path=roc_curve_path
)
