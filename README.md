# Semiconductor Manufacturing Yield Analysis

## Data Description
Source:
https://archive.ics.uci.edu/dataset/179/secom

* Data comes from semiconductor manufacturing. 
* ~ 600 **unlabeled** features, including pass/fail yield.
* ~ 1600 rows, indexed by a timestamp. Each timestamp corresponds to a "production unit" per documentation.

**Goals and Strategy**

In this analysis we will train progressively improved random forest models to identify top yield predictors. 

## Data Preparation


```python
import pandas as pd
import polars as pl

# Read in the data
# Define schema to treat all 591 columns as Float64. 
# This is necessary because polars will infer i64 on some columns, 
# probably because the first values encountered are integers 
schema = {f"column_{i}": pl.Float64 for i in range(591)}

data = pl.read_csv(
    source = r'data/secom.data', 
    has_header = False,
    separator = ' ',
    null_values = 'NaN',
    schema_overrides = schema)
#data.tail()

# Read the indeces
indeces = pl.read_csv(
    source = r'data/secom_labels.data',
    separator = ' ',
    has_header = False,
    new_columns = ['pass_fail','timestamp'],
    try_parse_dates=True
).select(['timestamp','pass_fail'])
# Select here just reorders the columns for display aesthetics
# indeces.tail()

# Will use hstack to stitch the files together as relation is implicit. Normally would join on explicit shared column. 
df = indeces.hstack(data)
df.to_pandas().tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>pass_fail</th>
      <th>column_1</th>
      <th>column_2</th>
      <th>column_3</th>
      <th>column_4</th>
      <th>column_5</th>
      <th>column_6</th>
      <th>column_7</th>
      <th>column_8</th>
      <th>...</th>
      <th>column_581</th>
      <th>column_582</th>
      <th>column_583</th>
      <th>column_584</th>
      <th>column_585</th>
      <th>column_586</th>
      <th>column_587</th>
      <th>column_588</th>
      <th>column_589</th>
      <th>column_590</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1562</th>
      <td>2008-10-16 15:13:00</td>
      <td>-1</td>
      <td>2899.41</td>
      <td>2464.36</td>
      <td>2179.7333</td>
      <td>3085.3781</td>
      <td>1.4843</td>
      <td>100.0</td>
      <td>82.2467</td>
      <td>0.1248</td>
      <td>...</td>
      <td>0.0047</td>
      <td>203.1720</td>
      <td>0.4988</td>
      <td>0.0143</td>
      <td>0.0039</td>
      <td>2.8669</td>
      <td>0.0068</td>
      <td>0.0138</td>
      <td>0.0047</td>
      <td>203.1720</td>
    </tr>
    <tr>
      <th>1563</th>
      <td>2008-10-16 20:49:00</td>
      <td>-1</td>
      <td>3052.31</td>
      <td>2522.55</td>
      <td>2198.5667</td>
      <td>1124.6595</td>
      <td>0.8763</td>
      <td>100.0</td>
      <td>98.4689</td>
      <td>0.1205</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.4975</td>
      <td>0.0131</td>
      <td>0.0036</td>
      <td>2.6238</td>
      <td>0.0068</td>
      <td>0.0138</td>
      <td>0.0047</td>
      <td>203.1720</td>
    </tr>
    <tr>
      <th>1564</th>
      <td>2008-10-17 05:26:00</td>
      <td>-1</td>
      <td>2978.81</td>
      <td>2379.78</td>
      <td>2206.3000</td>
      <td>1110.4967</td>
      <td>0.8236</td>
      <td>100.0</td>
      <td>99.4122</td>
      <td>0.1208</td>
      <td>...</td>
      <td>0.0025</td>
      <td>43.5231</td>
      <td>0.4987</td>
      <td>0.0153</td>
      <td>0.0041</td>
      <td>3.0590</td>
      <td>0.0197</td>
      <td>0.0086</td>
      <td>0.0025</td>
      <td>43.5231</td>
    </tr>
    <tr>
      <th>1565</th>
      <td>2008-10-17 06:01:00</td>
      <td>-1</td>
      <td>2894.92</td>
      <td>2532.01</td>
      <td>2177.0333</td>
      <td>1183.7287</td>
      <td>1.5726</td>
      <td>100.0</td>
      <td>98.7978</td>
      <td>0.1213</td>
      <td>...</td>
      <td>0.0075</td>
      <td>93.4941</td>
      <td>0.5004</td>
      <td>0.0178</td>
      <td>0.0038</td>
      <td>3.5662</td>
      <td>0.0262</td>
      <td>0.0245</td>
      <td>0.0075</td>
      <td>93.4941</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>2008-10-17 06:07:00</td>
      <td>-1</td>
      <td>2944.92</td>
      <td>2450.76</td>
      <td>2195.4444</td>
      <td>2914.1792</td>
      <td>1.5978</td>
      <td>100.0</td>
      <td>85.1011</td>
      <td>0.1235</td>
      <td>...</td>
      <td>0.0045</td>
      <td>137.7844</td>
      <td>0.4987</td>
      <td>0.0181</td>
      <td>0.0040</td>
      <td>3.6275</td>
      <td>0.0117</td>
      <td>0.0162</td>
      <td>0.0045</td>
      <td>137.7844</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 592 columns</p>
</div>




```python
# As sanity check let's count the failure rate
fails = indeces.select('pass_fail').filter( pl.col.pass_fail == 1 ).height

fail_rate = 100*fails/indeces.height
print(f"The failure rate is {round(fail_rate)}%")
```

    The failure rate is 7%
    

## Data Analysis
### Random Forest, first try


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X = data.to_numpy()
y = df['pass_fail'].to_numpy()

# Get feature names for importances
feature_names = df.drop(["pass_fail", "timestamp"]).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
f = model.fit(X_train, y_train)
```

After succesfully fitting the model we can extract the most important features.


```python
import pandas as pd
importances = pd.Series(model.feature_importances_, index=feature_names)
print('Top predictors')
print('-----------------------')
print(importances.sort_values(ascending=False).head(10))
```

    Top predictors
    -----------------------
    column_563    0.013259
    column_66     0.012310
    column_65     0.012131
    column_575    0.009252
    column_268    0.008815
    column_427    0.008118
    column_442    0.007570
    column_39     0.007448
    column_27     0.006392
    column_60     0.006293
    dtype: float64
    

As sanity check let us check the distribution of test sets to ensure a proper proportion of failures are present.


```python
# Class distribution
# Let's double check that the test and train sets are properly stratified.
print(pd.Series(y_train).value_counts(normalize=True))
print(pd.Series(y_test).value_counts(normalize=True))
```

    -1    0.933759
     1    0.066241
    Name: proportion, dtype: float64
    -1    0.933121
     1    0.066879
    Name: proportion, dtype: float64
    

Now let's evaluate the model. 


```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print( classification_report(y_test, y_pred) )
```

                  precision    recall  f1-score   support
    
              -1       0.93      1.00      0.97       293
               1       0.00      0.00      0.00        21
    
        accuracy                           0.93       314
       macro avg       0.47      0.50      0.48       314
    weighted avg       0.87      0.93      0.90       314
    
    

    C:\prog\WPy64-31241\python-3.12.4.amd64\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    C:\prog\WPy64-31241\python-3.12.4.amd64\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    C:\prog\WPy64-31241\python-3.12.4.amd64\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    

This model is very bad. The warnings come from the fact that zero failures were correctly identified.
* Precision (  True positive / (False positive + True positive) ) is **zero** for yield failures.
* Recall (  True positive / (False negative + True positive) is zero as well.

This means the features listed above may not be very important. Performance may improve by balancing the model training across pass/fail, to be explored next.

## Random Forest with "balanced" training
Here we'll use a version of the random forest algorithm that balances the fraction of failures in the training data fed to every tree in the ensemble 50/50.


```python
from sklearn.impute import SimpleImputer
from imblearn.ensemble import BalancedRandomForestClassifier

# Impute missing values -- in contrast to non-balanced function, this function cannot handle NaNs. 
# Hence we will impute with median. 
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train model
model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

# Predict with default threshold (0.5)
y_pred = model.predict(X_test_imputed)

# Classification report
print("\nClassification Report")
print(classification_report(y_test, y_pred, labels=[-1, 1], target_names=["Pass", "Fail"], zero_division=0))

# Feature importances
# This will now work because len(model.feature_importances_) == len(feature_names)
importances = pd.Series(model.feature_importances_, index=feature_names)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))
```

    
    Classification Report
                  precision    recall  f1-score   support
    
            Pass       0.95      0.91      0.93       293
            Fail       0.18      0.29      0.22        21
    
        accuracy                           0.87       314
       macro avg       0.56      0.60      0.57       314
    weighted avg       0.90      0.87      0.88       314
    
    
    Top 10 Feature Importances:
    column_104    0.013821
    column_34     0.011932
    column_65     0.011552
    column_130    0.009553
    column_478    0.008211
    column_342    0.007600
    column_60     0.007068
    column_82     0.006649
    column_206    0.006486
    column_181    0.006441
    dtype: float64
    

This is a marked improvement in that the model now correctly predicts 20% of failures, up from zero. Let's explore if hyperparameter tuning can further improve the model.
