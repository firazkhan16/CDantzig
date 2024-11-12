import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
website_data = pd.read_csv(r"walkthrough\500_Site_Info_Example.csv")
interaction_data = pd.read_csv(r"walkthrough\Page_View_Matrix_Example.csv")

X = website_data[['Cost', 'Pages', 'Clickthrough']]
y = np.ones(X.shape[0])  # Target vector of ones for equal budget across websites
y = np.full(X.shape[0], 1/500)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run Lasso regression to determine budget allocation coefficients
lasso = Lasso(alpha=0.01, positive=True)  # Adjust alpha as needed
lasso.fit(X_scaled, y)

# Retrieve the allocation coefficients
beta = lasso.coef_
beta_normalized = beta / np.sum(beta)  # Normalize to ensure it sums to 1

# Display the results
print("Normalized Budget Allocation Coefficients:", beta_normalized)