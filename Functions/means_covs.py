import numpy as np
from scipy.stats import norm

def generate_means_covs(num_classes, overlap_percentage=0.3, sigma=1.0, random_state=42):
    
    np.random.seed(random_state)
    
    means = []
    covs = []
    
    # חישוב המרחק בין התוחלות בהתבסס על אחוז החפיפה
    d = -2 * sigma * norm.ppf(overlap_percentage)


    # חלוקת זויות באופן שווה סביב מעגל כדי למקם את התוחלות בצורת מעגל
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    for i in range(num_classes):
        angle = angles[i]
        
        # התוחלת (mean) במחלקה ה-i
        mean = [d * np.cos(angle), d * np.sin(angle)]
        means.append(mean)

        # יצירת מטריצת קובריאנס שונה לכל ענן:
        # 1) var_x, var_y – שונות שונה לכל ציר
        # 2) rho – קורלציה רנדומלית בין הצירים
        var_x = sigma**2 * np.random.uniform(0.5, 1.5)
        var_y = sigma**2 * np.random.uniform(0.5, 1.5)
        
        # נגריל קורלציה בין -0.7 ל-0.7 לדוגמה (ניתן להחליף לטווח אחר כרצונכם)
        rho = np.random.uniform(-0.7, 0.7)
        
        # איברי השונות המשולבים (cov_xy) = rho * sqrt(var_x * var_y)
        cov_xy = rho * np.sqrt(var_x * var_y)
        
        # מרכיבים את מטריצת הקובריאנס
        cov_matrix = np.array([
            [var_x,  cov_xy],
            [cov_xy, var_y ]
        ])
        
        covs.append(cov_matrix.tolist())

    return means, covs