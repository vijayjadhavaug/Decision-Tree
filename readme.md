## hyperparameters for DT:

### 1. Critetion: gini or entropy. 
gini perfome better compared to entropy because entropy has log in it calculations

### 2. splitter: best/random --> 
random is not used frequently
splitting criteria is decided as follows: you check with every value by splitting it to know where is the maximum information gain. you can also decide the do the splitting randomly. the advantage is that it helps reduce overfiting.

you can take any feature and put some criteria and start splitting until purity [a distinct class] is achieved.

Splitting will continue only until one of the stopping conditions is met! 🚦

🌳 Stopping Conditions for a Decision Tree:
Purity is achieved (all 0s or all 1s in a node).
a. Max depth is reached (max_depth parameter).
b. Min samples per split is reached (min_samples_split parameter).
c. Min samples per leaf is reached (min_samples_leaf parameter).
d. Min impurity decrease is not enough (min_impurity_decrease parameter).

If purity is not achieved but one of the above conditions is met, splitting will stop, and the node will remain impure.

## 3. max_depth

default is None --> means you are not controlling the depth. chances of overfitting is high

if value is too small - under fitting
if value is too big - over fitting


## Q. Can we say that if the ENTROPY=0, then no further splitting. does it mean splitting will happen until entropy becomes zero?

When Entropy = 0 → No Further Splitting
Entropy measures impurity (uncertainty).
If Entropy = 0, it means the dataset is completely pure, meaning all instances belong to one class only (either all "Yes" or all "No").
Since there is no uncertainty left, no further splitting is required, and that node becomes a leaf node in a decision tree.

More the UNCERTAINITY, more is the ENTROPY. KNOWLEDGE = 1/ENTROPY
For a 2 class problem, MIN entropy = 0 [when you arrive at a single class - i.e pure] and MAX is = 1 [when you have same number of YES and NO in your dataset]
For more than 2 classes, the MIN entropy = 0 and MAX = > 1
You can use both log2 or loge to calculate ENTROPY.


![alt text](https://file%2B.vscode-resource.vscode-cdn.net/var/folders/h9/rdlk6cxd6t9blwl8t79q47cw0000gn/T/TemporaryItems/NSIRD_screencaptureui_7qHqdt/Screenshot%202025-02-07%20at%2011.13.25%E2%80%AFAM.png?version%3D1738907023368)

TIll now we saw onlt categorical variables:

# PART 1: ENTROPY
# ********************

## ENTROPY for CONTINUOUS Variables

when you draw an histogram or a KDE plot of the dataset, if the peak is LESS ENTROY is HIGH

when you split the data and if the ENTROPY is reduced, it means you have high information gain.

To calculate information Gain [IG] for a particular column : 

IG = H(Parent) - Weighted Avg * H(Children) where:

H(Parent) is the ENTROPY of the Parent - complete dataset.

When you do SPLIT, you get CHildren for which you calculate the Weighted AVG of the ENTROPY of all the Children

The calculations is done in following steps:
We assume a dataset having 5 columns and 1 tagetr column with classes. total records = 14

## STEP 1: Calculate Parent ENTROPY

this is calculated for the complete dataset  = Summation of - pi * log(pi) H(P) = 0.97

## STEP 2: Calculate ENTROPY for Children.
This is done by Grouping the dataset into parts based on a column. lets say we get 3 child datasets
Total records in dataset = 14. out of which 
sub-dataset 1 has [5 records]: 3 No and 2 Yes. Hence ENTROPY = 0.97
sub-dataset 2 has [4 records]: all 4 = Yes. Hence ENTROPY = 0 ==> This becomes a leaf node since E=0
sub-dataset 3 has [5 records]: 3 Yes and 2 No. Hence ENTROPY = 0.97

## STEP 3: Calculate Weighted ENTROPY for Children [3 sub-datasets above]

Weighted ENTROPY = 5/14 * 0.97 + 4/14 * 0 + 5/14 * 0.97
Hence Weighted ENTROPY of CHildren = 0.69

## STEP 4: Calculate Information Gain [IG]

IG = H(P) - {weighted Avg} * H(C)
IG = 0.97 - 0.69
IG = 0.28

Form these 4 steps, we found that the IG is 0.28 when you split the data on the basis of 1 column

## STEP 5: Calculate IG for all columns [1 col we already did]

Repeat the above 4 steps for the remaining columns.

Note: Outcome: whichever column has the highest IG [max decrease in ENTROPY], we will finally select that column to split the data.

## STEP 6: Find the IG recursively.
DT then applies the recursive greedy search algorithm in top bottom fashion to find the IG at every level of the tree.
Once a leaf node is reached at ENTROPY=0, splitting is stopped.


# PART 2: GINI
# ********************

GIni is very similar to ENTROPY, just the difference is in the formula.

Formula: G = 1 = (Py**2 + Pn**2)

Ex. 
1. if a dataset has only 1 class, mean the data us PURE, hence

E = G = 0

2. if dataset has same number of classes [ 5-Y and 5 - N], then

E = 1
G = 0.5

This means the max value for E = 1 and G = 0.5. The bell curve for gini is half of ENTROPY

SPlit of ENTROPY is bit balanced compared ti Gini. ENTROPY provides more balanced splits whereas Gini overfits

Gini uses SQUARE and ENTROPY uses LOG, hence Gini is faster in computation
## **************

### Here’s a comparison table for different models based on their suitability for large datasets with many numerical columns:

<table>
  <tr>
    <th>Model</th>
    <th>Handles Large Datasets?</th>
    <th>Overfitting Risk?</th>
    <th>Computational Efficiency</th>
    <th>Handles Numerical Features Well?</th>
    <th>Interpretability</th>
    <th>Best Use Case</th>
  </tr>
  <tr>
    <td><b>Decision Tree (DT)</b></td>
    <td>❌ Not ideal</td>
    <td>🔴 High</td>
    <td>🟡 Moderate</td>
    <td>🟢 Yes</td>
    <td>🟢 Easy to interpret</td>
    <td>Small datasets, explainability</td>
  </tr>
  <tr>
    <td><b>Random Forest (RF)</b></td>
    <td>🟡 Moderate</td>
    <td>🟡 Medium (reduces overfitting)</td>
    <td>🟡 Slower than DT</td>
    <td>🟢 Yes</td>
    <td>🟡 Moderate</td>
    <td>General-purpose, balanced performance</td>
  </tr>
  <tr>
    <td><b>XGBoost / LightGBM</b></td>
    <td>🟢 Excellent</td>
    <td>🟢 Low</td>
    <td>🟢 Fast (handles large datasets well)</td>
    <td>🟢 Yes</td>
    <td>🔴 Hard to interpret</td>
    <td>Large datasets, structured data</td>
  </tr>
  <tr>
    <td><b>CatBoost</b></td>
    <td>🟢 Excellent</td>
    <td>🟢 Low</td>
    <td>🟢 Fast</td>
    <td>🟢 Yes</td>
    <td>🔴 Hard to interpret</td>
    <td>Large datasets, categorical + numerical mix</td>
  </tr>
  <tr>
    <td><b>Neural Networks (DNNs)</b></td>
    <td>🟢 Excellent</td>
    <td>🟡 Can overfit</td>
    <td>🔴 High computational cost</td>
    <td>🟢 Yes</td>
    <td>🔴 Black-box model</td>
    <td>Very large datasets, deep learning tasks</td>
  </tr>
</table>


