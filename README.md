# Customer-Segmentation-with-K-Means-Clustering
This is my first data science project where I used K-Means clustering to segment 200 mall customers into 5 distinct groups based on their Annual Income and Spending Score. The goal was to help a mall business understand their customers better so they can make smarter marketing decisions.



## Dataset
- **Name:** Mall Customers Dataset
- **Size:** 200 rows, 5 columns

| Column | Description |
|---|---|
| CustomerID | Unique ID for each customer |
| Gender | Male or Female |
| Age | Age of the customer |
| Annual Income (k$) | How much the customer earns per year |
| Spending Score (1-100) | Score assigned by the mall based on spending behavior |

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
```

---

## Steps

### Step 1 — Load the Dataset

```python
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.tail())
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

**What we found:**
- 200 customers and 5 columns
- No missing values anywhere
- Gender is the only text column, everything else is a number

---

### Step 2 — Explore and Visualize the Data

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(df['Age'], ax=axes[0], kde=True, color='blue')
axes[0].set_title('Age Distribution')

sns.histplot(df['Annual Income (k$)'], ax=axes[1], kde=True, color='green')
axes[1].set_title('Annual Income Distribution')

sns.histplot(df['Spending Score (1-100)'], ax=axes[2], kde=True, color='red')
axes[2].set_title('Spending Score Distribution')

plt.tight_layout()
plt.show()
```

```python
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df, palette='Set2')
plt.title('Gender vs Spending Score')
plt.show()
```

**What we found:**

| Chart | Finding |
|---|---|
| Age Distribution | Most customers are between 25 and 35 years old |
| Annual Income Distribution | Most customers earn between $60k and $80k |
| Spending Score Distribution | Customers spend very differently from each other |
| Gender vs Spending Score | Both genders have similar spending behavior |

---

### Step 3 — Select Features for Clustering

```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print(X.head())
```

We selected only **Annual Income** and **Spending Score** because these two columns best describe customer behavior.

---

### Step 4 — Find the Optimal Number of Clusters

**Elbow Method:**

```python
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='blue')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

**Silhouette Score:**

```python
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'Number of clusters: {i}, Silhouette Score: {score:.4f}')
```

**Results:**

| Number of Clusters | Silhouette Score |
|---|---|
| 2 | 0.3956 |
| 3 | 0.4676 |
| 4 | 0.4938 |
| **5** | **0.5539 ✅ Highest** |
| 6 | 0.5128 |
| 7 | 0.5017 |
| 8 | 0.4963 |
| 9 | 0.4559 |
| 10 | 0.4426 |

Both the Elbow Method and Silhouette Score agreed that **5 is the optimal number of clusters.**

---

### Step 5 — Build the K-Means Model

```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
print(df['Cluster'].value_counts())
```

**Cluster sizes:**

| Cluster | Number of Customers |
|---|---|
| Cluster 0 | 81 customers |
| Cluster 1 | 39 customers |
| Cluster 3 | 35 customers |
| Cluster 4 | 23 customers |
| Cluster 2 | 22 customers |

---

### Step 6 — Visualize the Clusters

```python
plt.figure(figsize=(10, 6))

sns.scatterplot(x=df['Annual Income (k$)'],
                y=df['Spending Score (1-100)'],
                hue=df['Cluster'],
                palette='Set1',
                s=100)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

---

## Cluster Interpretation

| Cluster | Income | Spending | Description |
|---|---|---|---|
| 🔴 Cluster 0 | Average | Average | The largest group with 81 customers. These are normal everyday shoppers who earn and spend moderately |
| 🔵 Cluster 1 | High | High | These customers earn a lot and spend a lot. They are the most valuable customers to the mall |
| 🟢 Cluster 2 | Low | High | Very interesting group. They do not earn much but they still spend a lot. They might be young people who love shopping |
| 🟠 Cluster 3 | Low | Low | These customers do not earn much and do not spend much either. They are very careful with their money |
| 🟡 Cluster 4 | High | Low | The most surprising group. They earn a lot but barely spend anything. Rich but very conservative |

---

## Challenges
- Encountered a `NameError: name 'X' is not defined` because I ran the clustering cell before running the cell that created the variable `X`. Fixed by running all cells in the correct order from top to bottom.

---

## What I Learned
- How to explore and understand a dataset before building a model
- How to use the Elbow Method and Silhouette Score to find the optimal number of clusters
- How to build and visualize a K-Means clustering model
- How different customers behave differently based on income and spending

---

## Tools Used
- Python
- Google Colab
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

---

*This is my first data science project and part of my data science portfolio.*
