# Semiconductor Manufacturing Yield Analysis

**Data set**:
https://archive.ics.uci.edu/dataset/179/secom
* ~ 600 features, including pass/fail yield.
* ~ 1600 rows, indexed by a timestamp. Each timestamp corresponds to a "production unit" per documentation. 

## Data Prep
Data and indeces are in two separate file. Need to stitch.


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
).select(['timestamp','pass_fail']).with_columns( 
    pl.when( pl.col.pass_fail == -1).then(pl.lit('p')).otherwise(pl.lit('f')).alias('pass_fail')
        ) # pass fail encoding per documentation. 
# Select here just reorders the columns for display aesthetics
# indeces.tail()

# Will use hstack to stitch the files together as relation is implicit. 
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
<small>shape: (5, 592)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>pass_fail</th><th>column_1</th><th>column_2</th><th>column_3</th><th>column_4</th><th>column_5</th><th>column_6</th><th>column_7</th><th>column_8</th><th>column_9</th><th>column_10</th><th>column_11</th><th>column_12</th><th>column_13</th><th>column_14</th><th>column_15</th><th>column_16</th><th>column_17</th><th>column_18</th><th>column_19</th><th>column_20</th><th>column_21</th><th>column_22</th><th>column_23</th><th>column_24</th><th>column_25</th><th>column_26</th><th>column_27</th><th>column_28</th><th>column_29</th><th>column_30</th><th>column_31</th><th>column_32</th><th>column_33</th><th>column_34</th><th>column_35</th><th>&hellip;</th><th>column_554</th><th>column_555</th><th>column_556</th><th>column_557</th><th>column_558</th><th>column_559</th><th>column_560</th><th>column_561</th><th>column_562</th><th>column_563</th><th>column_564</th><th>column_565</th><th>column_566</th><th>column_567</th><th>column_568</th><th>column_569</th><th>column_570</th><th>column_571</th><th>column_572</th><th>column_573</th><th>column_574</th><th>column_575</th><th>column_576</th><th>column_577</th><th>column_578</th><th>column_579</th><th>column_580</th><th>column_581</th><th>column_582</th><th>column_583</th><th>column_584</th><th>column_585</th><th>column_586</th><th>column_587</th><th>column_588</th><th>column_589</th><th>column_590</th></tr><tr><td>datetime[μs]</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>&hellip;</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>2008-10-16 15:13:00</td><td>&quot;p&quot;</td><td>2899.41</td><td>2464.36</td><td>2179.7333</td><td>3085.3781</td><td>1.4843</td><td>100.0</td><td>82.2467</td><td>0.1248</td><td>1.3424</td><td>-0.0045</td><td>-0.0057</td><td>0.9579</td><td>203.9867</td><td>0.0</td><td>11.7692</td><td>419.3404</td><td>10.2397</td><td>0.9693</td><td>193.747</td><td>12.5373</td><td>1.4072</td><td>-5418.75</td><td>2608.0</td><td>-6228.25</td><td>356.0</td><td>1.2817</td><td>1.954</td><td>7.0793</td><td>71.1444</td><td>2.2222</td><td>0.1753</td><td>3.468</td><td>83.8405</td><td>8.7164</td><td>50.2482</td><td>&hellip;</td><td>8.5138</td><td>0.3141</td><td>85.1806</td><td>4.2063</td><td>1.0367</td><td>1.0972</td><td>0.3553</td><td>0.0929</td><td>32.3812</td><td>264.272</td><td>0.5671</td><td>4.98</td><td>0.0877</td><td>2.0902</td><td>0.0382</td><td>1.8844</td><td>15.4662</td><td>536.3418</td><td>2.0153</td><td>7.98</td><td>0.2363</td><td>2.6401</td><td>0.0785</td><td>1.4879</td><td>11.7256</td><td>0.0068</td><td>0.0138</td><td>0.0047</td><td>203.172</td><td>0.4988</td><td>0.0143</td><td>0.0039</td><td>2.8669</td><td>0.0068</td><td>0.0138</td><td>0.0047</td><td>203.172</td></tr><tr><td>2008-10-16 20:49:00</td><td>&quot;p&quot;</td><td>3052.31</td><td>2522.55</td><td>2198.5667</td><td>1124.6595</td><td>0.8763</td><td>100.0</td><td>98.4689</td><td>0.1205</td><td>1.4333</td><td>-0.0061</td><td>-0.0093</td><td>0.9618</td><td>204.0173</td><td>0.0</td><td>9.162</td><td>405.8178</td><td>10.2285</td><td>0.9696</td><td>193.7889</td><td>12.402</td><td>1.3949</td><td>-6408.75</td><td>2277.5</td><td>-3675.5</td><td>339.0</td><td>1.087</td><td>1.8023</td><td>5.1515</td><td>72.8444</td><td>2.0</td><td>0.1416</td><td>4.7088</td><td>84.0623</td><td>8.9607</td><td>50.2067</td><td>&hellip;</td><td>6.7381</td><td>0.5058</td><td>27.0176</td><td>3.6251</td><td>1.8156</td><td>0.9671</td><td>0.3105</td><td>0.0696</td><td>32.1048</td><td>266.832</td><td>0.6254</td><td>4.56</td><td>0.1308</td><td>1.742</td><td>0.0495</td><td>1.7089</td><td>20.9118</td><td>537.9264</td><td>2.1814</td><td>5.48</td><td>0.3891</td><td>1.9077</td><td>0.1213</td><td>1.0187</td><td>17.8379</td><td>null</td><td>null</td><td>null</td><td>null</td><td>0.4975</td><td>0.0131</td><td>0.0036</td><td>2.6238</td><td>0.0068</td><td>0.0138</td><td>0.0047</td><td>203.172</td></tr><tr><td>2008-10-17 05:26:00</td><td>&quot;p&quot;</td><td>2978.81</td><td>2379.78</td><td>2206.3</td><td>1110.4967</td><td>0.8236</td><td>100.0</td><td>99.4122</td><td>0.1208</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>12.4555</td><td>1.4256</td><td>-5153.25</td><td>2707.0</td><td>-4102.0</td><td>-1226.0</td><td>1.293</td><td>1.9435</td><td>7.2315</td><td>71.2667</td><td>2.2333</td><td>0.1659</td><td>3.4912</td><td>85.8638</td><td>8.1728</td><td>50.9333</td><td>&hellip;</td><td>7.0023</td><td>0.5605</td><td>74.1541</td><td>4.1095</td><td>2.0228</td><td>0.9718</td><td>0.1266</td><td>0.0332</td><td>13.0316</td><td>256.73</td><td>0.8209</td><td>11.09</td><td>0.2388</td><td>4.4128</td><td>0.0965</td><td>4.3197</td><td>29.0954</td><td>530.3709</td><td>2.3435</td><td>6.49</td><td>0.4154</td><td>2.176</td><td>0.1352</td><td>1.2237</td><td>17.7267</td><td>0.0197</td><td>0.0086</td><td>0.0025</td><td>43.5231</td><td>0.4987</td><td>0.0153</td><td>0.0041</td><td>3.059</td><td>0.0197</td><td>0.0086</td><td>0.0025</td><td>43.5231</td></tr><tr><td>2008-10-17 06:01:00</td><td>&quot;p&quot;</td><td>2894.92</td><td>2532.01</td><td>2177.0333</td><td>1183.7287</td><td>1.5726</td><td>100.0</td><td>98.7978</td><td>0.1213</td><td>1.4622</td><td>-0.0072</td><td>0.0032</td><td>0.9694</td><td>197.2448</td><td>0.0</td><td>9.7354</td><td>401.9153</td><td>9.863</td><td>0.974</td><td>187.3818</td><td>12.3937</td><td>1.3868</td><td>-5271.75</td><td>2676.5</td><td>-4001.5</td><td>394.75</td><td>1.2875</td><td>1.988</td><td>7.3255</td><td>70.5111</td><td>2.9667</td><td>0.2386</td><td>3.2803</td><td>84.5602</td><td>9.193</td><td>50.6547</td><td>&hellip;</td><td>6.7381</td><td>0.5058</td><td>27.0176</td><td>3.6251</td><td>1.8156</td><td>1.0108</td><td>0.192</td><td>0.0435</td><td>18.9966</td><td>264.272</td><td>0.5671</td><td>4.98</td><td>0.0877</td><td>2.0902</td><td>0.0382</td><td>1.8844</td><td>15.4662</td><td>534.3936</td><td>1.9098</td><td>9.13</td><td>0.3669</td><td>3.2524</td><td>0.104</td><td>1.7085</td><td>19.2104</td><td>0.0262</td><td>0.0245</td><td>0.0075</td><td>93.4941</td><td>0.5004</td><td>0.0178</td><td>0.0038</td><td>3.5662</td><td>0.0262</td><td>0.0245</td><td>0.0075</td><td>93.4941</td></tr><tr><td>2008-10-17 06:07:00</td><td>&quot;p&quot;</td><td>2944.92</td><td>2450.76</td><td>2195.4444</td><td>2914.1792</td><td>1.5978</td><td>100.0</td><td>85.1011</td><td>0.1235</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>12.479</td><td>1.4048</td><td>-5319.5</td><td>2668.0</td><td>-3951.75</td><td>-425.0</td><td>1.302</td><td>2.0085</td><td>7.3395</td><td>73.0667</td><td>2.5889</td><td>0.2021</td><td>3.386</td><td>83.3424</td><td>8.7786</td><td>50.194</td><td>&hellip;</td><td>6.7381</td><td>0.5058</td><td>27.0176</td><td>3.6251</td><td>1.8156</td><td>1.0827</td><td>0.2327</td><td>0.0678</td><td>21.4914</td><td>257.974</td><td>0.6193</td><td>8.42</td><td>0.1307</td><td>3.0894</td><td>0.0493</td><td>3.2639</td><td>21.1128</td><td>528.7918</td><td>2.0831</td><td>6.81</td><td>0.4774</td><td>2.2727</td><td>0.1495</td><td>1.2878</td><td>22.9183</td><td>0.0117</td><td>0.0162</td><td>0.0045</td><td>137.7844</td><td>0.4987</td><td>0.0181</td><td>0.004</td><td>3.6275</td><td>0.0117</td><td>0.0162</td><td>0.0045</td><td>137.7844</td></tr></tbody></table></div>




```python
# As sanity check let's count the failures
indeces.select('pass_fail').filter( pl.col.pass_fail == 'f' ).height
```




    104


