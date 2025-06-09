# Semiconductor Manufacturing Yield Analysis

**Data set**:
https://archive.ics.uci.edu/dataset/179/secom
* ~ 600 features, including pass/fail yield.
* ~ 1600 rows, indexed by a timestamp. Each timestamp corresponds to a "production unit" per documentation. 

## Data Preparation


```python
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
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 592)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>pass_fail</th><th>column_1</th><th>column_2</th><th>column_3</th><th>column_4</th><th>column_5</th><th>column_6</th><th>column_7</th><th>column_8</th><th>column_9</th><th>column_10</th><th>column_11</th><th>column_12</th><th>column_13</th><th>column_14</th><th>column_15</th><th>column_16</th><th>column_17</th><th>column_18</th><th>column_19</th><th>column_20</th><th>column_21</th><th>column_22</th><th>column_23</th><th>column_24</th><th>column_25</th><th>column_26</th><th>column_27</th><th>column_28</th><th>column_29</th><th>column_30</th><th>column_31</th><th>column_32</th><th>column_33</th><th>column_34</th><th>column_35</th><th>&hellip;</th><th>column_554</th><th>column_555</th><th>column_556</th><th>column_557</th><th>column_558</th><th>column_559</th><th>column_560</th><th>column_561</th><th>column_562</th><th>column_563</th><th>column_564</th><th>column_565</th><th>column_566</th><th>column_567</th><th>column_568</th><th>column_569</th><th>column_570</th><th>column_571</th><th>column_572</th><th>column_573</th><th>column_574</th><th>column_575</th><th>column_576</th><th>column_577</th><th>column_578</th><th>column_579</th><th>column_580</th><th>column_581</th><th>column_582</th><th>column_583</th><th>column_584</th><th>column_585</th><th>column_586</th><th>column_587</th><th>column_588</th><th>column_589</th><th>column_590</th></tr><tr><td>datetime[μs]</td><td>i64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>&hellip;</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>2008-10-16 15:13:00</td><td>-1</td><td>2899.41</td><td>2464.36</td><td>2179.7333</td><td>3085.3781</td><td>1.4843</td><td>100.0</td><td>82.2467</td><td>0.1248</td><td>1.3424</td><td>-0.0045</td><td>-0.0057</td><td>0.9579</td><td>203.9867</td><td>0.0</td><td>11.7692</td><td>419.3404</td><td>10.2397</td><td>0.9693</td><td>193.747</td><td>12.5373</td><td>1.4072</td><td>-5418.75</td><td>2608.0</td><td>-6228.25</td><td>356.0</td><td>1.2817</td><td>1.954</td><td>7.0793</td><td>71.1444</td><td>2.2222</td><td>0.1753</td><td>3.468</td><td>83.8405</td><td>8.7164</td><td>50.2482</td><td>&hellip;</td><td>8.5138</td><td>0.3141</td><td>85.1806</td><td>4.2063</td><td>1.0367</td><td>1.0972</td><td>0.3553</td><td>0.0929</td><td>32.3812</td><td>264.272</td><td>0.5671</td><td>4.98</td><td>0.0877</td><td>2.0902</td><td>0.0382</td><td>1.8844</td><td>15.4662</td><td>536.3418</td><td>2.0153</td><td>7.98</td><td>0.2363</td><td>2.6401</td><td>0.0785</td><td>1.4879</td><td>11.7256</td><td>0.0068</td><td>0.0138</td><td>0.0047</td><td>203.172</td><td>0.4988</td><td>0.0143</td><td>0.0039</td><td>2.8669</td><td>0.0068</td><td>0.0138</td><td>0.0047</td><td>203.172</td></tr><tr><td>2008-10-16 20:49:00</td><td>-1</td><td>3052.31</td><td>2522.55</td><td>2198.5667</td><td>1124.6595</td><td>0.8763</td><td>100.0</td><td>98.4689</td><td>0.1205</td><td>1.4333</td><td>-0.0061</td><td>-0.0093</td><td>0.9618</td><td>204.0173</td><td>0.0</td><td>9.162</td><td>405.8178</td><td>10.2285</td><td>0.9696</td><td>193.7889</td><td>12.402</td><td>1.3949</td><td>-6408.75</td><td>2277.5</td><td>-3675.5</td><td>339.0</td><td>1.087</td><td>1.8023</td><td>5.1515</td><td>72.8444</td><td>2.0</td><td>0.1416</td><td>4.7088</td><td>84.0623</td><td>8.9607</td><td>50.2067</td><td>&hellip;</td><td>6.7381</td><td>0.5058</td><td>27.0176</td><td>3.6251</td><td>1.8156</td><td>0.9671</td><td>0.3105</td><td>0.0696</td><td>32.1048</td><td>266.832</td><td>0.6254</td><td>4.56</td><td>0.1308</td><td>1.742</td><td>0.0495</td><td>1.7089</td><td>20.9118</td><td>537.9264</td><td>2.1814</td><td>5.48</td><td>0.3891</td><td>1.9077</td><td>0.1213</td><td>1.0187</td><td>17.8379</td><td>null</td><td>null</td><td>null</td><td>null</td><td>0.4975</td><td>0.0131</td><td>0.0036</td><td>2.6238</td><td>0.0068</td><td>0.0138</td><td>0.0047</td><td>203.172</td></tr><tr><td>2008-10-17 05:26:00</td><td>-1</td><td>2978.81</td><td>2379.78</td><td>2206.3</td><td>1110.4967</td><td>0.8236</td><td>100.0</td><td>99.4122</td><td>0.1208</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>12.4555</td><td>1.4256</td><td>-5153.25</td><td>2707.0</td><td>-4102.0</td><td>-1226.0</td><td>1.293</td><td>1.9435</td><td>7.2315</td><td>71.2667</td><td>2.2333</td><td>0.1659</td><td>3.4912</td><td>85.8638</td><td>8.1728</td><td>50.9333</td><td>&hellip;</td><td>7.0023</td><td>0.5605</td><td>74.1541</td><td>4.1095</td><td>2.0228</td><td>0.9718</td><td>0.1266</td><td>0.0332</td><td>13.0316</td><td>256.73</td><td>0.8209</td><td>11.09</td><td>0.2388</td><td>4.4128</td><td>0.0965</td><td>4.3197</td><td>29.0954</td><td>530.3709</td><td>2.3435</td><td>6.49</td><td>0.4154</td><td>2.176</td><td>0.1352</td><td>1.2237</td><td>17.7267</td><td>0.0197</td><td>0.0086</td><td>0.0025</td><td>43.5231</td><td>0.4987</td><td>0.0153</td><td>0.0041</td><td>3.059</td><td>0.0197</td><td>0.0086</td><td>0.0025</td><td>43.5231</td></tr><tr><td>2008-10-17 06:01:00</td><td>-1</td><td>2894.92</td><td>2532.01</td><td>2177.0333</td><td>1183.7287</td><td>1.5726</td><td>100.0</td><td>98.7978</td><td>0.1213</td><td>1.4622</td><td>-0.0072</td><td>0.0032</td><td>0.9694</td><td>197.2448</td><td>0.0</td><td>9.7354</td><td>401.9153</td><td>9.863</td><td>0.974</td><td>187.3818</td><td>12.3937</td><td>1.3868</td><td>-5271.75</td><td>2676.5</td><td>-4001.5</td><td>394.75</td><td>1.2875</td><td>1.988</td><td>7.3255</td><td>70.5111</td><td>2.9667</td><td>0.2386</td><td>3.2803</td><td>84.5602</td><td>9.193</td><td>50.6547</td><td>&hellip;</td><td>6.7381</td><td>0.5058</td><td>27.0176</td><td>3.6251</td><td>1.8156</td><td>1.0108</td><td>0.192</td><td>0.0435</td><td>18.9966</td><td>264.272</td><td>0.5671</td><td>4.98</td><td>0.0877</td><td>2.0902</td><td>0.0382</td><td>1.8844</td><td>15.4662</td><td>534.3936</td><td>1.9098</td><td>9.13</td><td>0.3669</td><td>3.2524</td><td>0.104</td><td>1.7085</td><td>19.2104</td><td>0.0262</td><td>0.0245</td><td>0.0075</td><td>93.4941</td><td>0.5004</td><td>0.0178</td><td>0.0038</td><td>3.5662</td><td>0.0262</td><td>0.0245</td><td>0.0075</td><td>93.4941</td></tr><tr><td>2008-10-17 06:07:00</td><td>-1</td><td>2944.92</td><td>2450.76</td><td>2195.4444</td><td>2914.1792</td><td>1.5978</td><td>100.0</td><td>85.1011</td><td>0.1235</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>12.479</td><td>1.4048</td><td>-5319.5</td><td>2668.0</td><td>-3951.75</td><td>-425.0</td><td>1.302</td><td>2.0085</td><td>7.3395</td><td>73.0667</td><td>2.5889</td><td>0.2021</td><td>3.386</td><td>83.3424</td><td>8.7786</td><td>50.194</td><td>&hellip;</td><td>6.7381</td><td>0.5058</td><td>27.0176</td><td>3.6251</td><td>1.8156</td><td>1.0827</td><td>0.2327</td><td>0.0678</td><td>21.4914</td><td>257.974</td><td>0.6193</td><td>8.42</td><td>0.1307</td><td>3.0894</td><td>0.0493</td><td>3.2639</td><td>21.1128</td><td>528.7918</td><td>2.0831</td><td>6.81</td><td>0.4774</td><td>2.2727</td><td>0.1495</td><td>1.2878</td><td>22.9183</td><td>0.0117</td><td>0.0162</td><td>0.0045</td><td>137.7844</td><td>0.4987</td><td>0.0181</td><td>0.004</td><td>3.6275</td><td>0.0117</td><td>0.0162</td><td>0.0045</td><td>137.7844</td></tr></tbody></table></div>




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
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_estimators&nbsp;</td>
            <td class="value">100</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">criterion&nbsp;</td>
            <td class="value">&#x27;gini&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_depth&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_samples_split&nbsp;</td>
            <td class="value">2</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_samples_leaf&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_weight_fraction_leaf&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">&#x27;sqrt&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_leaf_nodes&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_impurity_decrease&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('bootstrap',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">bootstrap&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('oob_score',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">oob_score&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">random_state&nbsp;</td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">warm_start&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('class_weight',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">class_weight&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ccp_alpha&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_samples',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_samples&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('monotonic_cst',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">monotonic_cst&nbsp;</td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>



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
    
    

    C:\prog\WPy64-31241\python-3.12.4.amd64\Lib\site-packages\sklearn\metrics\_classification.py:1706: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\prog\WPy64-31241\python-3.12.4.amd64\Lib\site-packages\sklearn\metrics\_classification.py:1706: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    C:\prog\WPy64-31241\python-3.12.4.amd64\Lib\site-packages\sklearn\metrics\_classification.py:1706: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
    

The warnings coming from the zeros in the fail class, discussed below. 

Overall the model is not very good at predicting failures:
* Precision (  True positive / (False positive + True positive) ) is **zero** for yield failures.
* Recall (  True positive / (False negative + True positive) is zero as well.

This means the features listed above may not be very important. Performance may improve by balancing the model training across pass/fail, to be explored next.
