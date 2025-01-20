from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc
)
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import cycle
import time
import draw_confidence_ellipse  # ודאי שהמודול שלך מוגדר כראוי

def test_model_fn(
      X,
    y,
    train_ratio,
    model,
    classes=None,
    decision_boundary_path=None,
    roc_curve_path=None
):
    
    
    # חלוקת הנתונים ל-train ו-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=42
    )
    
    
    # הגדרת מחלקות אם לא הוגדרו
    if classes is None:
        classes = sorted(list(set(y)))
    
    # מדידת זמן התחלה
    start_time = time.time()
    
    # אימון המודל
    model.fit(X_train, y_train)
    
    # מדידת זמן סיום
    end_time = time.time()
    training_time = end_time - start_time
    
    # חיזוי על סט הבדיקה
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # הדפסת צורות סט הבדיקה
    print(f"Test set size: {X_test.shape}, y_pred shape: {y_pred.shape}, y_pred_proba shape: {y_pred_proba.shape}")
    
    # הערכת המודל
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError as e:
        print(f"Error calculating AUC: {e}")
        auc_score = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    # הדפסת תוצאות
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    if auc_score is not None:
        print(f"AUC: {auc_score:.4f}")
    else:
        print("AUC: Not available")
    print(f"Training Time: {training_time:.2f} seconds")
    print("Confusion Matrix:")
    print(cm)
    
    # ציור גבולות החלטה ואליפסות ביטחון
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    
    # הגדרת תחום הציור
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # צביעת השטח לפי מחלקה חזויה
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.rainbow)
    
    # ציור הנקודות והאליפסות
    for idx, (label, color) in enumerate(zip(classes, colors)):
        X_class = X_test[y_pred == label]
        ax.scatter(
            X_class[:, 0], X_class[:, 1],
            label=f'Predicted Class {label}',
            alpha=0.6, edgecolor='k', color=color
        )
        
        if len(X_class) > 1:
            draw_confidence_ellipse.confidence_ellipse(
                X_class[:, 0], X_class[:, 1], ax, n_std=1,
                edgecolor=color, linewidth=2, linestyle='-'
            )
            draw_confidence_ellipse.confidence_ellipse(
                X_class[:, 0], X_class[:, 1], ax, n_std=2,
                edgecolor=color, linewidth=1, linestyle='--'
            )
            draw_confidence_ellipse.confidence_ellipse(
                X_class[:, 0], X_class[:, 1], ax, n_std=3,
                edgecolor=color, linewidth=1, linestyle=':'
            )
    
    # הגדרת כותרות ותוויות
    plt.title("Gaussian Clouds Predicted by Model with Decision Boundaries and Ellipses")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    
    # שמירת גרף גבולות ההחלטה למחשב
    plt.savefig(decision_boundary_path)
    print(f"Decision boundaries plot saved to {decision_boundary_path}")
    
    # הצגת הגרף
    plt.show()
    
    def plot_roc_curve(y_test, y_pred_proba, classes, num_classes, roc_curve_path):
     if num_classes == 2:
        # במקרה של 2 מחלקות
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc_value = auc(fpr, tpr)

        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'Binary ROC (area = {roc_auc_value:0.2f})', lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Binary ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(roc_curve_path)
        print(f"ROC curve plot saved to {roc_curve_path}")
        plt.show()

     else:
        # במקרה של יותר משתי מחלקות
        y_test_binarized = label_binarize(y_test, classes=classes)
        n_classes = y_test_binarized.shape[1]

        fpr = {}
        tpr = {}
        roc_auc_dict = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
        roc_auc_dict["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(8,6))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {i} (area = {roc_auc_dict[i]:0.2f})')

        plt.plot(fpr["micro"], tpr["micro"], 'b--', lw=2,
                 label=f'Micro-average ROC (area = {roc_auc_dict["micro"]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(roc_curve_path)
        print(f"ROC curve plot saved to {roc_curve_path}")
        plt.show()
    
    # חישוב וציור ROC
    n_classes = len(classes)  # לפי מה שהגדרת num_classes מראש בחוץ
    plot_roc_curve(y_test, y_pred_proba, classes, n_classes, roc_curve_path)


    return X_test, y_test, y_pred, y_pred_proba
