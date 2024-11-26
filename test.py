###
import pandas as pd
import numpy as np

# Load the datasets
page_view_matrix = pd.read_csv(r"walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0)
site_info = pd.read_csv(r"walkthrough/500_Site_Info_Example.csv", header=0)

# Extract necessary columns
cost = site_info['Cost']  # Cost per impression (CPM)
traffic = site_info['Pages']  # Total website visits (traffic)
clickthrough = site_info['Clickthrough']  # CTR

# Calculate gamma (efficiency metric for reach)
gamma = 1 / (cost * traffic)

# Replace NaN values in page view matrix with 0
page_view_matrix.fillna(0, inplace=True)

# Normalize gamma to ensure it sums to 1 (optional, for proportional allocation)
gamma_normalized = gamma / gamma.sum()

# Initialize variables for budget steps
num_steps = 251
step_size = 0.02  # Incremental budget steps
budgets = np.arange(0, num_steps * step_size, step_size)  # Budget levels
reach_elmso = []

# Compute reach for each budget step
for budget in budgets:
    # Allocate budget proportional to gamma
    allocation = gamma_normalized * budget  # Budget allocation for each website

    # Compute reach: Fraction of users exposed at least once
    exposure = page_view_matrix.values @ allocation.values  # Total exposure per user
    reach = (exposure > 0).mean()  # Fraction of users reached
    reach_elmso.append(reach)

# Convert results to a DataFrame
results = pd.DataFrame({
    "Budget": budgets,
    "Reach": reach_elmso
})

# Display results
print(results)

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.plot(results["Budget"], results["Reach"], label="Reach (ELMSO)", color="blue")
plt.xlabel("Budget (in millions)")
plt.ylabel("Reach")
plt.title("Reach vs. Budget (ELMSO)")
plt.legend()
plt.show()
###


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Load synthetic dataset
page_view_matrix = pd.read_csv(r"walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0)
site_info = pd.read_csv(r"walkthrough/500_Site_Info_Example.csv", header=0)

# Feature matrix (X) and target vector (y)
X = page_view_matrix.values  # Rows: Users, Columns: Websites
y = np.ones(X.shape[0])  # Equal allocation baseline

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configure and fit Lasso
lasso = Lasso(alpha=5.0, positive=True, tol=0.1, max_iter=100000)
lasso.fit(X_scaled, y)

# Extract coefficients
coefficients = lasso.coef_

# Normalize coefficients to represent proportional budget allocation
beta_normalized = coefficients / np.sum(coefficients)

# Identify selected websites (non-zero coefficients)
non_zero_coefficients = (coefficients != 0).sum()
selected_websites = site_info.loc[coefficients != 0, ['Site_Name', 'Cost', 'Pages', 'Clickthrough']]

# Display results
print("Converged Lasso Regression")
print(f"Number of Non-Zero Coefficients: {non_zero_coefficients}")
print("Normalized Coefficients (Proportional Allocation):")
print(beta_normalized)
print("\nSelected Websites for Budget Allocation:")
print(selected_websites)
