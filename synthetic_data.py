import numpy as np
import pandas as pd

# Parameters
n_users = 10000
age_groups = ["<18", "18-24", "25-34", "35-44", "45-54", ">55"]
income_brackets = ["<30k", "30-60k", "60-100k", ">100k"]
regions = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
]  # Assume Region are continents excluding Antarctica

# Random demographic assignments
np.random.seed(42)
user_data = pd.DataFrame(
    {
        "user_id": range(n_users),
        "age_group": np.random.choice(age_groups, n_users),
        "income_bracket": np.random.choice(income_brackets, n_users),
        "region": np.random.choice(regions, n_users),
    }
)


n_websites = 100
website_categories = [
    "Entertainment",
    "Community",
    "Fileshare",
    "SocialNetwork",
    "Travel",
    "Portal",
    "GeneralNews",
    "Gaming",
    "Retail",
    "Newspaper",
    "Email",
    "Service",
    "Information",
    "OnlineShop",
    "Sports",
    "Photos",
]  # Based on metadata provided by walkthrough on PAC
# Generate CPM values from a normal distribution, clamped to a reasonable range (e.g., 1 to 20)
cpm_values = np.clip(
    np.random.normal(loc=10, scale=4, size=len(website_categories)), 1, 20
)
cpm_values = np.round(cpm_values, 2)
# Generate CTR values from a uniform distribution in a typical range for CTR (e.g., 0.01 to 0.05)
ctr_values = np.random.uniform(0.01, 0.05, size=len(website_categories))

# Random category assignment
website_data = pd.DataFrame(
    {
        "website_id": range(n_websites),
        "category": np.random.choice(website_categories, n_websites),
        "cpm": np.random.choice(cpm_values, n_websites),
        "ctr": np.random.choice(ctr_values, n_websites),
    }
)

# Randomly generate the number of visits and page views per user per website
n_visits = np.random.poisson(
    5, (n_users, n_websites)
)  # Average 5 visits per user per website
page_views = np.random.poisson(
    3, (n_users, n_websites)
)  # Average 3 page views per visit

# Exposure probability: CPM and traffic volume influence reach
exposure_prob = (
    np.random.rand(n_users, n_websites) * 0.1
)  # Probability that a user sees an ad

# Clicks (based on exposure and CTR)
click_prob = exposure_prob * website_data["ctr"].values
clicks = (np.random.rand(n_users, n_websites) < click_prob).astype(int)


# Flatten matrices into a DataFrame for analysis
interaction_data = pd.DataFrame(
    {
        "user_id": np.repeat(user_data["user_id"], n_websites),
        "website_id": np.tile(website_data["website_id"], n_users),
        "visits": n_visits.flatten(),
        "page_views": page_views.flatten(),
        "exposure_prob": exposure_prob.flatten(),
        "clicks": clicks.flatten(),
    }
)

# Merge in user demographics and website metadata
interaction_data = interaction_data.merge(user_data, on="user_id", how="left")
interaction_data = interaction_data.merge(website_data, on="website_id", how="left")

# Preview the dataset
interaction_data.head()

# Calculate reach: percentage of users exposed to an ad on each website
reach = interaction_data.groupby("website_id")["exposure_prob"].mean()

# Calculate click rate for each website
click_rate = (
    interaction_data.groupby("website_id")["clicks"].sum()
    / interaction_data.groupby("website_id")["exposure_prob"].sum()
)

# Add reach and click_rate to the website_data for analysis
website_data = website_data.assign(reach=reach.values, click_rate=click_rate.values)

