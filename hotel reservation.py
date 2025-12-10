# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("Hotel Reservations renamed.csv")
df.head()

# %%
df.columns

# %%
print(df.meal_plan.value_counts())

# %%
# Visualize distribution of meal_plan
plt.bar(df.meal_plan.value_counts().index, df.meal_plan.value_counts())

# %%
# Convert meal_plan categories to ordered numerical values
df.meal_plan.replace(['Not Selected', "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"],
                     [0, 1, 2, 3], inplace=True)

# %%
df.head()

# %%
# Visualize distribution of room_type
plt.bar(df.room_type.value_counts().index, df.room_type.value_counts())
plt.xticks(df.room_type.value_counts().index, rotation=45)

# %%
# Convert room_type into multiple dummy (0–1) variables
dummies = pd.get_dummies(df['room_type'])
dummies.head()

# %%
df = pd.concat([df, dummies.iloc[:, :6]], axis=1)
df.head()

# %%
# Convert remaining categorical variables into binary numerical values
df.status = (df.status == "Not_Canceled").astype(int)
df.segment_type = (df.segment_type == "Online").astype(int)
df[['status', 'segment_type']]

# %%
# Remove irrelevant fields
df.drop(columns=["ID", "room_type", "year", "date"], inplace=True)
df.head()

# %%
# Compute correlation matrix
cor_mat = df.corr()
cor_mat

# %%
# Heatmap of correlation matrix
plt.figure(figsize=(12,7), dpi=100)
sns.heatmap(cor_mat, vmax=1, vmin=-1, center=0, square=True)

# %% [markdown]
# # Clustering

# %%
df.columns

# %%
# Remove target variable to prepare clustering data
df2 = df.drop(columns=['status'])
df2.head()

# %%
# Standardize features
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
data_s = standardscaler.fit_transform(df2)
data_s[:2]

# %%
from sklearn.cluster import KMeans
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(data_s)

# Plot cluster sizes
plt.hist(labels, bins=range(n_clusters + 1))
plt.title('Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Customers')
plt.show()

# %%
# Elbow Method to evaluate cluster count
sse = []
cluster_list = range(1, 10)
for i in cluster_list:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_s)
    sse.append(kmeans.inertia_)

plt.plot(cluster_list, sse)
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.show()

# %%
# Silhouette Score evaluation
from sklearn.metrics import silhouette_score
s = []
cluster_list = range(2, 6)
for i in cluster_list:
    kmeans = KMeans(n_clusters=i)
    s.append(silhouette_score(data_s, kmeans.fit_predict(data_s)))

plt.bar(cluster_list, s)
plt.xlabel('Number of clusters', fontsize=10)
plt.ylabel('Silhouette Score', fontsize=10)
plt.show()

# %%
# Fit final model with k = 4
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(data_s)
df["cluster"] = labels

# Plot cluster distribution
plt.hist(labels, bins=range(5))
plt.title('Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Customers')
plt.show()

# %%
# Visualize cluster centers
plt.subplots(figsize=(10, 10))
centers = kmeans.cluster_centers_
idx = np.arange(len(df2.columns))

plt.bar(idx, centers[0], color='b', width=0.25, tick_label=df2.columns)
plt.bar(idx + 0.25, centers[1], color='g', width=0.25)
plt.bar(idx + 0.50, centers[2], color='r', width=0.25)
plt.bar(idx + 0.75, centers[3], color='y', width=0.25)
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# # Classification

# %%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# %%
# Train-test split (default ratio ~ 0.75 / 0.25)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['status']), df.status)

# %%
# Create classifier and parameter grid
clf = RandomForestClassifier(random_state=0)
param_grid = {'n_estimators': list(range(10, 110, 10))}

# Hyperparameter tuning using cross-validation
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# %%
# Plot CV scores for each n_estimators value
scores = grid_search.cv_results_['mean_test_score']
plt.plot(param_grid['n_estimators'], scores, 'o-', color="blue")
plt.xlabel('n_estimators')
plt.ylabel('Score')
plt.title('Cross-validation scores')
plt.show()

# %%
# Retrain model with best parameter (40)
rf_clf = RandomForestClassifier(n_estimators=40)
rf_clf.fit(X_train, y_train)

# Test accuracy
y_pred = rf_clf.predict(X_test)
accuracy_score(y_pred, y_test)

# %%
# Feature importance scores
rf_clf.feature_importances_

# %%
# Visualize feature importances
plt.figure(figsize=(10, 8))
plt.bar(X_test.columns, rf_clf.feature_importances_)
plt.xticks(X_test.columns, rotation=90)

# %% [markdown]
# # Visualization

# %%
plt.figure(figsize=(10, 6))
sns.boxenplot(x="status", y="lead_time", data=df)

# %%
plt.figure(figsize=(10, 6))
sns.boxenplot(x="status", y="special_requests", data=df)

# %%
plt.figure(figsize=(10, 6))
sns.countplot(y="status", hue="month", data=df)

# %%
plt.figure(figsize=(10, 6))
sns.countplot(y="status", hue="segment_type", data=df)
