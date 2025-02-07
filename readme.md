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