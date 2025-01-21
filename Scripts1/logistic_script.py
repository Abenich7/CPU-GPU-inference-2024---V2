import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

num_samples = 10000
factor = 0.5
noise = 0.05
train_ratio=0.8

# יצירת הדאטה
X, y = make_circles(n_samples=num_samples, noise=noise, factor=factor, random_state=42)


# חלוקת הנתונים
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

# סטנדרטיזציה של הנתונים
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_proba_log_reg = log_reg.predict_proba(X_test)

# הערכת Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
mse_log_reg = mean_squared_error(y_test, y_pred_log_reg)
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:, 1])

print("Logistic Regression:")
print(f"Accuracy: {accuracy_log_reg:.4f}")
print(f"MSE: {mse_log_reg:.4f}")
print(f"Confusion Matrix:\n{cm_log_reg}")
print(f"AUC: {auc_log_reg:.4f}\n")


# Linear Regression as Classifier
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
threshold = 0.5
y_pred_lin_reg_class = (y_pred_lin_reg >= threshold).astype(int)

# הערכת Linear Regression
accuracy_lin_reg = accuracy_score(y_test, y_pred_lin_reg_class)
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg_class)
cm_lin_reg = confusion_matrix(y_test, y_pred_lin_reg_class)
auc_lin_reg = roc_auc_score(y_test, y_pred_lin_reg)

print("Linear Regression (as Classifier):")
print(f"Accuracy: {accuracy_lin_reg:.4f}")
print(f"MSE: {mse_lin_reg:.4f}")
print(f"Confusion Matrix:\n{cm_lin_reg}")
print(f"AUC: {auc_lin_reg:.4f}\n")

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5)
plt.title('make_circles Dataset')
plt.show()



def plot_decision_boundary(model, X, y, title, ax):
    # הגדרת רשת לציור גבולות ההחלטה
    h = 0.02  # רזולוציית הרשת
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # תחזיות על כל נקודות הרשת
    if isinstance(model, LogisticRegression):
        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    elif isinstance(model, LinearRegression):
        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = (Z >= threshold).astype(int)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    # ציור גבולות ההחלטה
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    # ציור הנקודות
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='bwr', alpha=0.5, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# ציור הדאטה והחיזויים
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# הדאטה המקורית
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='bwr', alpha=0.5, ax=axes[0], legend=False)
axes[0].set_title('Original Data')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# חיזוי Logistic Regression
plot_decision_boundary(log_reg, X_test, y_test, 'Logistic Regression Decision Boundary', axes[1])

# חיזוי Linear Regression
plot_decision_boundary(lin_reg, X_test, y_test, 'Linear Regression Decision Boundary', axes[2])

plt.tight_layout()
plt.show()