# Experiment: Python Data Processing and Visualization
## Exploratory Data Analysis Based on Restaurant Tip Dataset

This experiment uses the restaurant tip dataset `tips.csv` to learn basic methods of Python data processing and visualization.

The dataset contains fields such as total consumption amount, tip amount, customer gender, smoking status, day of the week, dining time, and party size.

**Key contents of this experiment:**
1. Load and inspect data using pandas.
2. Identify and handle missing values.
3. Identify and handle abnormal or inconsistent values in categorical fields.
4. Conduct descriptive statistics and grouped statistics.
5. Perform visual analysis with various charts.
6. Understand the basic workflow of exploratory data analysis.

---

## 1. Import Libraries
This experiment mainly uses:
- **pandas**: Data reading, cleaning and statistical analysis
- **numpy**: Numerical computation
- **matplotlib**: Basic plotting
- **seaborn**: More aesthetic and statistical visualization

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")

# Fix Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 2. Load Data
Place the data file `tips.csv` in the same directory as the notebook.

```python
df = pd.read_csv("tips.csv")
df.head()
```

|  | total_bill | tip | sex | smoker | day | time | size |
|---|---|---|---|---|---|---|---|
| 0 | 16.99 | 1.01 | Female | No | Sun | Dinner | 2 |
| 1 | 10.34 | 1.66 | Male | No | Sun | Dinner | 3 |
| 2 | 21.01 | 3.50 | Male | No | Sun | Dinner | 3 |
| 3 | 23.68 | 3.31 | Male | No | Dinner | 2 |
| 4 | 24.59 | 3.61 | Female | No | Sun | Dinner | 4 |

---

## 3. Inspect Basic Data Information
Before formal analysis, check the number of rows, columns, field types, and basic statistics.

```python
# Check data size
df.shape
```
Output: `(244, 7)`

```python
# Check field names, non-null counts and data types
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   total_bill  244 non-null    float64
 1   tip         243 non-null    float64
 2   sex         244 non-null    object
 3   smoker      244 non-null    object
 4   day         244 non-null    object
 5   time        243 non-null    object
 6   size        244 non-null    int64
dtypes: float64(2), int64(1), object(4)
memory usage: 13.5+ KB
```

```python
# Descriptive statistics for numerical fields
df.describe()
```

|  | total_bill | tip | size |
|---|---|---|---|
| count | 244.000000 | 243.000000 | 244.000000 |
| mean | 19.785943 | 3.006502 | 2.569672 |
| std | 8.902412 | 1.380506 | 0.951100 |
| min | 3.070000 | 1.000000 | 1.000000 |
| 25% | 13.347500 | 2.000000 | 2.000000 |
| 50% | 17.795000 | 2.920000 | 2.000000 |
| 75% | 24.127500 | 3.575000 | 3.000000 |
| max | 50.810000 | 10.000000 | 6.000000 |

```python
# Descriptive statistics for categorical fields
df.describe(include='object')
```

|  | sex | smoker | day | time |
|---|---|---|---|---|
| count | 244 | 244 | 244 | 243 |
| unique | 4 | 2 | 4 | 2 |
| top | Male | No | Sat | Dinner |
| freq | 156 | 151 | 87 | 175 |

---

## 4. Field Explanation

| Field Name | Meaning | Type |
|---|---|---|
| total_bill | Total consumption amount | Numerical |
| tip | Tip amount | Numerical |
| sex | Customer gender | Categorical |
| smoker | Smoking status | Categorical |
| day | Day of the week | Categorical |
| time | Dining time (Lunch/Dinner) | Categorical |
| size | Party size | Numerical/Discrete |

---

## 5. Missing Value Inspection and Handling
Missing values are common in data analysis.

**Strategy in this experiment**: Delete rows with any missing values.

Note: In real research, the handling method depends on data size, missing ratio, and business meaning. This experiment uses the simplest deletion method for beginners.

```python
# Count missing values per field
df.isnull().sum()
```
```
total_bill    0
tip           1
sex           0
smoker        0
day           0
time          1
size          0
dtype: int64
```

```python
# View samples with missing values
df[df.isnull().any(axis=1)]
```

| total_bill | tip | sex | smoker | day | time | size |
|---|---|---|---|---|---|---|
| 3.07 | NaN | Female | Yes | Sat | Dinner | 1 |
| 22.75 | 3.25 | Female | No | Fri | NaN | 2 |

```python
# Delete rows with missing values
df_clean = df.dropna()
print("Data size before deletion:", df.shape)
print("Data size after deletion:", df_clean.shape)
```
Output:
```
Data size before deletion: (244, 7)
Data size after deletion: (242, 7)
```

```python
# Recheck missing values
df_clean.isnull().sum()
```
All fields show 0.

---

## 6. Clean Inconsistent Categorical Values
The standard values for the `sex` field should be `Male`/`Female`.

Due to manual entry, there may be lowercase `male`/`female`. Python treats them as different categories, which affects statistics and visualization.

```python
# Check original values
df_clean['sex'].value_counts()
```
```
Male      156
Female     84
male        1
female      1
Name: count, dtype: int64
```

```python
# Standardize: capitalize first letter
df_clean = df_clean.copy()
df_clean['sex'] = df_clean['sex'].str.strip().str.capitalize()

# Check cleaned values
df_clean['sex'].value_counts()
```
```
Male      157
Female     85
Name: count, dtype: int64
```

```python
# Verify no invalid values remain
valid_sex = ['Male', 'Female']
df_clean[~df_clean['sex'].isin(valid_sex)]
```
Empty result.

---

## 7. Inspect Other Categorical Fields
```python
categorical_cols = ['sex', 'smoker', 'day', 'time']
for col in categorical_cols:
    print(f"Field: {col}")
    print(df_clean[col].value_counts())
    print("-" * 40)
```

Output:
```
Field: sex
Male      157
Female     85
Name: count, dtype: int64
----------------------------------------
Field: smoker
No     150
Yes     92
Name: count, dtype: int64
----------------------------------------
Field: day
Sat     86
Sun     76
Thur    62
Fri     18
Name: count, dtype: int64
----------------------------------------
Field: time
Dinner    174
Lunch      68
Name: count, dtype: int64
```

---

## 8. Create New Feature: Tip Rate
Define `tip_rate = tip / total_bill` (tip percentage of total bill).

```python
df_clean['tip_rate'] = df_clean['tip'] / df_clean['total_bill']
df_clean.head()
```

|  | total_bill | tip | sex | smoker | day | time | size | tip_rate |
|---|---|---|---|---|---|---|---|---|
| 0 | 16.99 | 1.01 | Female | No | Sun | Dinner | 2 | 0.059447 |
| 1 | 10.34 | 1.66 | Male | No | Sun | Dinner | 3 | 0.160542 |
| 2 | 21.01 | 3.50 | Male | No | Sun | Dinner | 3 | 0.166587 |
| 3 | 23.68 | 3.31 | Male | No | Dinner | 2 | 0.139780 |
| 4 | 24.59 | 3.61 | Female | No | Sun | Dinner | 4 | 0.146808 |

```python
# Statistics for new feature
df_clean[['total_bill', 'tip', 'tip_rate', 'size']].describe()
```

|  | total_bill | tip | tip_rate | size |
|---|---|---|---|---|
| count | 242.000000 | 242.000000 | 242.000000 | 242.000000 |
| mean | 19.842769 | 3.005496 | 0.160195 | 2.578512 |
| std | 8.871951 | 1.383278 | 0.060384 | 0.948920 |
| min | 5.750000 | 1.000000 | 0.035638 | 1.000000 |
| 25% | 13.375000 | 2.000000 | 0.128538 | 2.000000 |
| 50% | 17.795000 | 2.900000 | 0.154770 | 2.000000 |
| 75% | 24.222500 | 3.587500 | 0.191022 | 3.000000 |
| max | 50.810000 | 10.000000 | 0.710345 | 6.000000 |

---

## 9. Univariate Visualization
Univariate analysis focuses on the distribution of a single variable.

Common plots: count plot, pie chart, histogram, box plot.

### 9.1 Customer Gender: Count Plot
```python
sns.countplot(data=df_clean, x='sex')
plt.title("Customer Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
```

### 9.2 Customer Gender: Pie Chart
```python
sex_counts = df_clean['sex'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Customer Gender Proportion")
plt.show()
```

### 9.3 Smoking Status: Pie Chart
```python
smoker_counts = df_clean['smoker'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Smoking Status Proportion")
plt.show()
```

### 9.4 Dining Count by Day
```python
day_order = ['Thur', 'Fri', 'Sat', 'Sun']
sns.countplot(data=df_clean, x='day', order=day_order)
plt.title("Dining Count by Day")
plt.xlabel("Day")
plt.ylabel("Count")
plt.show()
```

### 9.5 Total Bill Distribution
```python
sns.histplot(data=df_clean, x='total_bill', bins=20, kde=True)
plt.title("Total Bill Distribution")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")
plt.show()
```

### 9.6 Tip Amount Distribution
```python
sns.histplot(data=df_clean, x='tip', bins=20, kde=True)
plt.title("Tip Amount Distribution")
plt.xlabel("Tip Amount")
plt.ylabel("Frequency")
plt.show()
```

### 9.7 Tip Rate Distribution
```python
sns.histplot(data=df_clean, x='tip_rate', bins=20, kde=True)
plt.title("Tip Rate Distribution")
plt.xlabel("Tip Rate")
plt.ylabel("Frequency")
plt.show()
```

---

## 10. Grouped Statistical Analysis
Compare numerical values across different categories.

```python
# By gender
df_clean.groupby('sex')[['total_bill', 'tip', 'tip_rate']].mean()
```

|  | total_bill | tip | tip_rate |
|---|---|---|---|
| Female | 18.178000 | 2.850118 | 0.164895 |
| Male | 20.744076 | 3.089618 | 0.157651 |

```python
# By dining time
df_clean.groupby('time')[['total_bill', 'tip', 'tip_rate']].mean()
```

|  | total_bill | tip | tip_rate |
|---|---|---|---|
| Dinner | 20.887816 | 3.113908 | 0.158658 |
| Lunch | 17.168676 | 2.728088 | 0.164128 |

```python
# By day
df_clean.groupby('day')[['total_bill', 'tip', 'tip_rate']].mean().reindex(day_order)
```

|  | total_bill | tip | tip_rate |
|---|---|---|---|
| Thur | 17.682742 | 2.771452 | 0.161276 |
| Fri | 16.840556 | 2.706111 | 0.171416 |
| Sat | 20.643372 | 3.016279 | 0.151145 |
| Sun | 21.410000 | 3.255132 | 0.166897 |

```python
# By smoking status
df_clean.groupby('smoker')[['total_bill', 'tip', 'tip_rate']].mean()
```

|  | total_bill | tip | tip_rate |
|---|---|---|---|
| No | 19.164533 | 2.990133 | 0.159438 |
| Yes | 20.948587 | 3.030543 | 0.161429 |

---

## 11. Grouped Visualization
### 11.1 Average Tip by Day
```python
sns.barplot(data=df_clean, x='day', y='tip', order=day_order)
plt.title("Average Tip by Day")
plt.xlabel("Day")
plt.ylabel("Average Tip")
plt.show()
```

### 11.2 Tip Distribution: Lunch vs Dinner (Box Plot)
```python
sns.boxplot(data=df_clean, x='time', y='tip')
plt.title("Tip Distribution: Lunch vs Dinner")
plt.xlabel("Dining Time")
plt.ylabel("Tip")
plt.show()
```

### 11.3 Tip Rate by Day (Box Plot)
```python
sns.boxplot(data=df_clean, x='day', y='tip_rate', order=day_order)
plt.title("Tip Rate Distribution by Day")
plt.xlabel("Day")
plt.ylabel("Tip Rate")
plt.show()
```

### 11.4 Tip Rate: Lunch vs Dinner (Violin Plot)
```python
sns.violinplot(data=df_clean, x='time', y='tip_rate')
plt.title("Tip Rate Distribution: Lunch vs Dinner")
plt.xlabel("Dining Time")
plt.ylabel("Tip Rate")
plt.show()
```

### 11.5 Total Bill vs Tip (Scatter Plot)
```python
sns.scatterplot(data=df_clean, x='total_bill', y='tip', hue='time', style='sex')
plt.title("Total Bill vs Tip Amount")
plt.xlabel("Total Bill")
plt.ylabel("Tip Amount")
plt.show()
```

### 11.6 Total Bill vs Tip with Trendline
```python
sns.regplot(data=df_clean, x='total_bill', y='tip')
plt.title("Total Bill vs Tip: Linear Trend")
plt.xlabel("Total Bill")
plt.ylabel("Tip Amount")
plt.show()
```

### 11.7 Party Size vs Total Bill
```python
sns.scatterplot(data=df_clean, x='size', y='total_bill', hue='time')
plt.title("Party Size vs Total Bill")
plt.xlabel("Party Size")
plt.ylabel("Total Bill")
plt.show()
```

### 11.8 Average Tip by Party Size
```python
sns.barplot(data=df_clean, x='size', y='tip')
plt.title("Average Tip by Party Size")
plt.xlabel("Party Size")
plt.ylabel("Average Tip")
plt.show()
```

---

## 12. Cross Tabulation & Heatmap
### 12.1 Day × Dining Time
```python
cross_day_time = pd.crosstab(df_clean['day'], df_clean['time'])
sns.heatmap(cross_day_time, annot=True, fmt='d', cmap='Blues')
plt.title("Day vs Dining Time Count")
plt.xlabel("Dining Time")
plt.ylabel("Day")
plt.show()
```

### 12.2 Gender × Smoking Status
```python
cross_sex_smoker = pd.crosstab(df_clean['sex'], df_clean['smoker'])
sns.heatmap(cross_sex_smoker, annot=True, fmt='d', cmap='Greens')
plt.title("Gender vs Smoking Status Count")
plt.xlabel("Smoking Status")
plt.ylabel("Gender")
plt.show()
```

---

## 13. Correlation Analysis
Correlation coefficient range: **-1 to 1**
- Close to 1: Strong positive correlation
- Close to -1: Strong negative correlation
- Close to 0: Weak linear correlation

```python
numeric_cols = ['total_bill', 'tip', 'size', 'tip_rate']
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Numeric Variables")
plt.show()
```

---

## 14. Outlier Observation (Box Plot)
```python
sns.boxplot(data=df_clean[['total_bill', 'tip', 'tip_rate']])
plt.title("Box Plot of Numeric Variables")
plt.show()
```

---

## 15. Experiment Summary
We completed a full exploratory data analysis workflow:
1. Load data
2. Inspect data structure
3. Check and remove missing values
4. Clean inconsistent categorical values
5. Create new analysis features
6. Conduct descriptive and grouped statistics
7. Visualize data with various charts
8. Analyze variable relationships
