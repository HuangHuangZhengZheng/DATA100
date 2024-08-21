```python
# Initialize Otter
import otter
grader = otter.Notebook("lab14.ipynb")
```

# Ungraded Lab 14: Clustering

In this lab you will explore K-Means, Agglomerative Clustering, and Spectral Clustering. Spectral Clustering is out of scope for Spring 2022.[ref.](https://github.com/MartinLwx/Data100-Fall-2021-UCB/blob/main/Labs/lab15.ipynb)

**Note: This is an ungraded assignment. There is no Gradescope submission for this assignment.** As this is a bonus and ungraded assignment, there will also be more limited staff office hours devoted to this ungraded homework. We may prioritize students who have other questions.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster

# more readable exceptions
%pip install --quiet iwut
%load_ext iwut
%wut on
```

    Note: you may need to restart the kernel to use updated packages.
    


In the first part of this lab, we work with three different toy datasets, all with different clustering characteristics. In the second part, we explore a real-world dataset from the World Bank.

<br/><br/><br/>

<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

## Toy Data 1: Balanced Clusters

Let us begin with a toy dataset with three groups that are completely separated with the variables given. There are the same number of points per group and the same variance within each group.


```python
# just run this cell
np.random.seed(1337)

c1 = np.random.normal(size = (25, 2))
c2 = np.array([2, 8]) + np.random.normal(size = (25, 2))
c3 = np.array([8, 4]) + np.random.normal(size = (25, 2))

x1 = np.vstack((c1, c2, c3))

sns.scatterplot(x = x1[:, 0], y = x1[:, 1]);
```


    
![png](lab14_files/lab14_4_0.png)
    


Below, we create a `cluster.KMeans` object ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)) which implements the K-Means algorithm.


```python
# just run this cell
kmeans = cluster.KMeans(n_clusters = 3, random_state = 42).fit(x1)
sns.scatterplot(x = x1[:, 0], y = x1[:, 1], hue = kmeans.labels_)
sns.scatterplot(x = kmeans.cluster_centers_[:, 0], y = kmeans.cluster_centers_[:, 1], color = 'blue', marker = 'x', s = 300, linewidth = 5);
```


    
![png](lab14_files/lab14_6_0.png)
    


We observe that K-Means is able to accurately pick out the three initial clusters. 

<br/><br/>

<hr style="border: 1px solid #fdb515;" />

## Question 1: Initial Centers

In the previous example, the K-Means algorithm was able to accurately find the three initial clusters. However, changing the starting centers for K-Means can change the final clusters that K-Means gives us. Change the initial centers to the points `[0, 1]`, `[1, 1]`, and `[2, 2]`; and fit a `cluster.KMeans` object ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)) called `kmeans_q1` on the toy dataset from the previous example. Keep the `random_state` parameter as 42 and the `n_clusters` parameter as 3.

**Hint:** You will need to change the `init` and `n_init` parameters in `cluster.KMeans`.

<!--
BEGIN QUESTION
name: q1
-->


```python
kmeans_q1 = cluster.KMeans(n_clusters = 3, random_state = 42, init=[[0,1],[1,1],[2,2]], n_init=1).fit(x1)
```


```python
grader.check("q1")
```




<p><strong><pre style='display: inline;'>q1</pre></strong> passed! 🍀</p>



Running the K-Means algorithm with these centers gives us a different result from before, and this particular run of K-Means was unable to accurately find the three initial clusters.


```python
sns.scatterplot(x = x1[:, 0], y = x1[:, 1], hue = kmeans_q1.labels_)
sns.scatterplot(x = kmeans_q1.cluster_centers_[:, 0], y = kmeans_q1.cluster_centers_[:, 1], color = 'blue', marker = 'x', s = 300, linewidth = 5);
```


    
![png](lab14_files/lab14_12_0.png)
    


<br/><br/><br/>

<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

## Toy Data 2: Clusters of Different Sizes

Sometimes, K-Means will have a difficult time finding the "correct" clusters even with ideal starting centers. For example, consider the data below.


```python
# just run this cell
np.random.seed(1337)

c1 = 0.5 * np.random.normal(size = (25, 2))
c2 = np.array([10, 10]) + 3 * np.random.normal(size = (475, 2))

x2 = np.vstack((c1, c2))

sns.scatterplot(x = x2[:, 0], y = x2[:, 1]);
```


    
![png](lab14_files/lab14_14_0.png)
    


There are two groups of different sizes in two different senses: **variability** (i.e., spread) and **number of datapoints**. The smaller group has both smaller variability and has fewer datapoints, and the larger of the two groups is more diffuse and populated.

<br/><br/>

<hr style="border: 1px solid #fdb515;" />

## Question 2

### Question 2a: K-Means

Fit a `cluster.KMeans` object ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)) called `kmeans_q2a` on the dataset above with two clusters and a `random_state` parameter of 42.

<!--
BEGIN QUESTION
name: q2a
-->


```python
kmeans_q2a = cluster.KMeans(n_clusters=2, random_state=42).fit(x2)
```


```python
grader.check("q2a")
```




<p><strong><pre style='display: inline;'>q2a</pre></strong> passed! 🍀</p>



<br/>

(For notational simplicity we will call the initial cluster on the bottom left $A$ and the initial cluster on the top right $B$. We will call the bottom left cluster found by K-Means as cluster $a$ and the top right cluster found by K-Means as cluster $b$.) 

As seen below, K-Means is unable to find the two intial clusters because cluster $A$ includes points from cluster $B$. Recall that K-Means attempts to minimize inertia (惯性？), so it makes sense that points in the bottom left of cluster $B$ would prefer to be in cluster $A$ rather than cluster $B$. If these points were in cluster $B$ instead, then the resulting cluster assignments would have a larger distortion(歪曲).


```python
# just run this cell
sns.scatterplot(x = x2[:, 0], y = x2[:, 1], hue = kmeans_q2a.labels_)
sns.scatterplot(x = kmeans_q2a.cluster_centers_[:, 0], y = kmeans_q2a.cluster_centers_[:, 1], color = 'red', marker = 'x', s = 300, linewidth = 5);
```


    
![png](lab14_files/lab14_20_0.png)
    


<br/>

---

### Agglomerative Clustering: The Linkage Criterion

It turns out agglomerative clustering works better for this task, as long as we choose the right definition of distance between two clusters. *Recall that agglomerative clustering starts with every data point in its own cluster and iteratively joins the two closest clusters until there are $k$ clusters remaining.* However, the "distance" between two clusters is ambiguous. 

In lecture, we used the maximum distance between a point in the first cluster and a point in the second as this notion of distance, but there are other ways to define the distance between two clusters. 

Our choice of definition for the distance is sometimes called the "linkage criterion." We will discuss three linkage criteria, each of which is a different definition of "distance" between two clusters:

- **Complete linkage** considers the distance between two clusters as the **maximum** distance between a point in the first cluster and a point in the second. This is what you will see in Lecture 26.
- **Single linkage** considers the distance between two clusters as the **minimum** distance between a point in the first cluster and a point in the second.
- **Average linkage** considers the distance between two clusters as the **average** distance between a point in the first cluster and a point in the second.

Below, we fit a `cluster.AgglomerativeClustering` object ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)) called `agg_complete` on the dataset above with two clusters, using the **complete linkage criterion**.


```python
# just run this cell
agg_complete = cluster.AgglomerativeClustering(n_clusters = 2, linkage = 'complete').fit(x2)
```

Below we visualize the results:


```python
# just run this cell
sns.scatterplot(x = x2[:, 0], y = x2[:, 1], hue = agg_complete.labels_);
```


    
![png](lab14_files/lab14_24_0.png)
    


It looks like complete linkage agglomerative clustering has the same issue as K-Means! The bottom left cluster found by complete linkage agglomerative clustering includes points from the top right cluster. However, we can remedy this by picking a different linkage criterion.

<br/>

---

### Question 2b: Agglomerative Clustering

Now, use the **single linkage criterion** to fit a `cluster.AgglomerativeClustering` object called `agg_single` on the dataset above with two clusters.

<!--
BEGIN QUESTION
name: q2b
-->


```python
agg_single = cluster.AgglomerativeClustering(n_clusters = 2, linkage = 'single').fit(x2)
```


```python
grader.check("q2b")
```




<p><strong><pre style='display: inline;'>q2b</pre></strong> passed! 🌟</p>



Finally, we see that single linkage agglomerative clustering is able to find the two initial clusters.


```python
sns.scatterplot(x = x2[:, 0], y = x2[:, 1], hue = agg_single.labels_);
```


    
![png](lab14_files/lab14_30_0.png)
    


You might be curious why single linkage "works" while complete linkage does not in this scenario; we will leave this as an exercise for students who are interested.

<br/><br/><br/>

<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

## Toy Data 3: Oddly Shaped Clusters

Another example when k-means fails is when the clusters have odd shapes. For example, look at the following dataset.


```python
np.random.seed(100)

data = np.random.normal(0, 7, size = (1000, 2))
lengths = np.linalg.norm(data, axis = 1, ord = 2)
x3 = data[(lengths < 2) | ((lengths > 5) & (lengths < 7)) | ((lengths > 11) & (lengths < 15))]

sns.scatterplot(x = x3[:, 0], y = x3[:, 1]);
```


    
![png](lab14_files/lab14_33_0.png)
    


Looking at this data, we might say there are 3 clusters, corresponding to each of the 3 concentric circles, with the same center. However, k-means will fail.


```python
kmeans_q3 = cluster.KMeans(n_clusters = 3, random_state = 42).fit(x3)
sns.scatterplot(x = x3[:, 0], y = x3[:, 1], hue = kmeans_q3.labels_)
sns.scatterplot(x = kmeans_q3.cluster_centers_[:, 0], y = kmeans_q3.cluster_centers_[:, 1], color = 'red', marker = 'x', s = 300, linewidth = 5);
```


    
![png](lab14_files/lab14_35_0.png)
    


<br/><br/>

<hr style="border: 1px solid #fdb515;" />

## (Bonus) Question 3: Spectral Clustering

(Note in Spring 2022 we did not go over Spectral Clustering. Spectral Clustering is out of scope for exams.) 

Let's try spectral clustering instead. 

In the cell below, create and fit a `cluster.SpectralClustering` object ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)), and assign it to `spectral`. Use 3 clusters, and make sure you set `affinity` to `"nearest_neighbors"` and a `random_state` of 10.

**Note:** Ignore any warnings about the graph not being fully connected.

<!--
BEGIN QUESTION
name: q3
-->


```python
spectral = cluster.SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=10).fit(x3)
```

    d:\miniconda3\envs\ds100\Lib\site-packages\sklearn\manifold\_spectral_embedding.py:329: UserWarning: Graph is not fully connected, spectral embedding may not work as expected.
      warnings.warn(
    


```python
grader.check("q3")
```




<p><strong><pre style='display: inline;'>q3</pre></strong> passed! ✨</p>



Below, we see that spectral clustering is able to find the three rings, when k-means does not.


```python
sns.scatterplot(x = x3[:, 0], y = x3[:, 1], hue = spectral.labels_);
```


    
![png](lab14_files/lab14_40_0.png)
    


<br/><br/><br/>

<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

## The World Bank Dataset

In the previous three questions, we looked at clustering on two dimensional datasets. However, we can easily use clustering on data which have more than two dimensions. For this, let us turn to a World Bank dataset, containing various features for the world's countries.

This data comes from https://databank.worldbank.org/source/world-development-indicators#.



```python
world_bank_data = pd.read_csv("world_bank_data.csv", index_col = 'country')
world_bank_data.head(5)
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
      <th>Age dependency ratio (% of working-age population)</th>
      <th>Age dependency ratio, old (% of working-age population)</th>
      <th>Age dependency ratio, young (% of working-age population)</th>
      <th>Bird species, threatened</th>
      <th>Business extent of disclosure index (0=less disclosure to 10=more disclosure)</th>
      <th>Contributing family workers, female (% of female employment) (modeled ILO estimate)</th>
      <th>Contributing family workers, male (% of male employment) (modeled ILO estimate)</th>
      <th>Contributing family workers, total (% of total employment) (modeled ILO estimate)</th>
      <th>Cost of business start-up procedures (% of GNI per capita)</th>
      <th>Cost of business start-up procedures, female (% of GNI per capita)</th>
      <th>...</th>
      <th>Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)</th>
      <th>Urban population</th>
      <th>Urban population (% of total population)</th>
      <th>Urban population growth (annual %)</th>
      <th>Vulnerable employment, female (% of female employment) (modeled ILO estimate)</th>
      <th>Vulnerable employment, male (% of male employment) (modeled ILO estimate)</th>
      <th>Vulnerable employment, total (% of total employment) (modeled ILO estimate)</th>
      <th>Wage and salaried workers, female (% of female employment) (modeled ILO estimate)</th>
      <th>Wage and salaried workers, male (% of male employment) (modeled ILO estimate)</th>
      <th>Wage and salaried workers, total (% of total employment) (modeled ILO estimate)</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Algeria</th>
      <td>57.508032</td>
      <td>10.021442</td>
      <td>47.486590</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>2.720000</td>
      <td>1.836</td>
      <td>1.978000</td>
      <td>0.0</td>
      <td>11.8</td>
      <td>...</td>
      <td>29.952999</td>
      <td>30670086.0</td>
      <td>72.629</td>
      <td>2.804996</td>
      <td>24.337001</td>
      <td>27.227001</td>
      <td>26.762000</td>
      <td>73.734001</td>
      <td>68.160004</td>
      <td>69.056000</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>84.077656</td>
      <td>4.758273</td>
      <td>79.319383</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>71.780998</td>
      <td>9.606</td>
      <td>31.577999</td>
      <td>0.0</td>
      <td>6.4</td>
      <td>...</td>
      <td>2.639000</td>
      <td>9477100.0</td>
      <td>25.495</td>
      <td>3.350383</td>
      <td>95.573997</td>
      <td>85.993001</td>
      <td>89.378998</td>
      <td>4.282000</td>
      <td>13.292000</td>
      <td>10.108000</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>45.810037</td>
      <td>20.041214</td>
      <td>25.768823</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>37.987000</td>
      <td>20.795</td>
      <td>28.076000</td>
      <td>0.0</td>
      <td>11.3</td>
      <td>...</td>
      <td>30.979000</td>
      <td>1728969.0</td>
      <td>60.319</td>
      <td>1.317162</td>
      <td>54.663000</td>
      <td>54.994001</td>
      <td>54.854000</td>
      <td>44.320999</td>
      <td>41.542999</td>
      <td>42.720001</td>
    </tr>
    <tr>
      <th>American Samoa</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>48339.0</td>
      <td>87.153</td>
      <td>-0.299516</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>67813.0</td>
      <td>88.062</td>
      <td>-0.092859</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 209 columns</p>
</div>



There are some missing values. For the sake of convenience and of keeping the lab short, we will fill them all with zeros. 


```python
world_bank_data = world_bank_data.fillna(0)
```

Like with PCA, it sometimes makes sense to center and scale our data so that features with higher variance don't dominate the analysis. For example, without standardization, statistics like population will completely dominate features like "percent of total population that live in urban areas." This is because the range of populations is on the order of billions, whereas percentages are always between 0 and 100. The ultimate effect is that many of our columns are not really considered by our clustering algorithm.


<br/><br/>

<hr style="border: 1px solid #fdb515;" />


## Question 4

Below, fit a `cluster.KMeans` object called `kmeans_q4` with four clusters and a `random_state` parameter of 42.

Make sure you should use a centered and scaled version of the world bank data. By centered and scaled we mean that the mean in each column should be zero and the variance should be 1.

<!--
BEGIN QUESTION
name: q4
-->


```python
world_bank_data = (world_bank_data - world_bank_data.mean(axis=0)) / world_bank_data.std(axis=0)
kmeans_q4 = cluster.KMeans(n_clusters = 4, 
                           random_state = 42).fit(world_bank_data)
```


```python
sorted(np.unique(kmeans_q4.labels_, return_counts = True)[1])
```




    [np.int64(3), np.int64(23), np.int64(85), np.int64(106)]



below is very interesting, note that in ref. all tests passed. ---> maybe Cluster itself changed?


```python
grader.check("q4")
```




<p><strong style='color: red;'><pre style='display: inline;'>q4</pre> results:</strong></p><p><strong><pre style='display: inline;'>q4 - 1</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        sorted(np.unique(kmeans_q4.labels_, return_counts = True)[1]) == [3, 25, 90, 99]
    Expecting:
        True
    **********************************************************************
    Line 1, in q4 0
    Failed example:
        sorted(np.unique(kmeans_q4.labels_, return_counts = True)[1]) == [3, 25, 90, 99]
    Expected:
        True
    Got:
        False
</pre>



Looking at these new clusters, we see that they seem to correspond to:

0: Very small countries.

1: Developed countries.

2: Less developed countries.

3: Huge countries.


```python
# just run this cell

labeled_world_bank_data_q4 = pd.Series(kmeans_q4.labels_, name = "cluster", index  = world_bank_data.index).to_frame()

for c in range(4):
    print(f">>> Cluster {c}:")
    print(list(labeled_world_bank_data_q4.query(f'cluster == {c}').index))
    print()
```

    >>> Cluster 0:
    ['Afghanistan', 'Angola', 'Bangladesh', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Botswana', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.', "Cote d'Ivoire", 'Djibouti', 'Ecuador', 'Egypt, Arab Rep.', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Fiji', 'Gabon', 'Gambia, The', 'Ghana', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Indonesia', 'Iraq', 'Kenya', 'Kiribati', 'Kyrgyz Republic', 'Lao PDR', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Micronesia, Fed. Sts.', 'Mongolia', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria', 'Pakistan', 'Papua New Guinea', 'Paraguay', 'Philippines', 'Rwanda', 'Samoa', 'Sao Tome and Principe', 'Senegal', 'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Sudan', 'Sudan', 'Syrian Arab Republic', 'Tajikistan', 'Tanzania', 'Timor-Leste', 'Togo', 'Tonga', 'Uganda', 'Uzbekistan', 'Vanuatu', 'Venezuela, RB', 'Vietnam', 'West Bank and Gaza', 'Yemen, Rep.', 'Zambia', 'Zimbabwe']
    
    >>> Cluster 1:
    ['China', 'India', 'United States']
    
    >>> Cluster 2:
    ['American Samoa', 'Andorra', 'Bermuda', 'British Virgin Islands', 'Cayman Islands', 'Dominica', 'Faroe Islands', 'Gibraltar', 'Greenland', 'Isle of Man', 'Kosovo', 'Liechtenstein', 'Marshall Islands', 'Monaco', 'Nauru', 'Northern Mariana Islands', 'Palau', 'San Marino', 'Sint Maarten (Dutch part)', 'St. Kitts and Nevis', 'St. Martin (French part)', 'Turks and Caicos Islands', 'Tuvalu']
    
    >>> Cluster 3:
    ['Algeria', 'Albania', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The', 'Bahrain', 'Barbados', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Brunei Darussalam', 'Bulgaria', 'Cabo Verde', 'Canada', 'Channel Islands', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Denmark', 'Dominican Republic', 'El Salvador', 'Estonia', 'Finland', 'France', 'French Polynesia', 'Georgia', 'Germany', 'Greece', 'Grenada', 'Guam', 'Hong Kong SAR, China', 'Hungary', 'Iceland', 'Iran, Islamic Rep.', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Korea, Dem. People’s Rep.', 'Korea, Rep.', 'Kuwait', 'Latvia', 'Lebanon', 'Libya', 'Lithuania', 'Luxembourg', 'Macao SAR, China', 'Malaysia', 'Maldives', 'Malta', 'Mauritius', 'Mexico', 'Moldova', 'Montenegro', 'Netherlands', 'New Caledonia', 'New Zealand', 'North Macedonia', 'Norway', 'Oman', 'Panama', 'Peru', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russian Federation', 'Saudi Arabia', 'Serbia', 'Seychelles', 'Singapore', 'Slovak Republic', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'St. Lucia', 'St. Vincent and the Grenadines', 'Suriname', 'Sweden', 'Switzerland', 'Thailand', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'Uruguay', 'Virgin Islands (U.S.)']
    
    

# Congratulations! You finished the lab!

---


