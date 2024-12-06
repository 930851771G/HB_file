# %% [markdown]
# # 1.导入库，读入数据。

# %%
#导入库：
import os
# import jovian
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
# import opendatasets as od
import plotly.express as px
import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# %%
#读入数据到df里面(dataframe结构)
train_df = pd.read_csv('D:\\Users\\gxy\\OneDrive\\directory\\YJY_BG\\Big_homework\\data\\data.csv')
train_df.head()
#每一列的含义、结构 在另外一个ipynb里面有解释。

# %%
#看一下train_df的结构：
train_df
#shape:76767 X 61

# %%
#查看数据信息：
train_df.info()

# %%
#查看数据的其他信息：
train_df.describe()

# %% [markdown]
# # 2.接下来开始做数据处理和特征工程
# 

# %%
#观察一下train_df:
train_df

# %% [markdown]
# **选择前60w条数据作为训练数据、60万到70万之间的数据作为测试数据**
# 
# （也可以之后选择随机划分）因为数据都是按照Bond_id进行排好序的，所以最好还是打乱一点，训练的效果会更好。

# %% [markdown]
# 

# %% [markdown]
# ## 2.1形成训练数据和测试数据

# %%
#借助 dataframe 的iloc进行数据的切片
X_train = train_df.iloc[:60000,:36].copy() 

X_test = train_df.iloc[60000:70000,:36].copy() 

# %%
X_train

# %%
X_test

# %% [markdown]
# 观察数据里面所有的列之后得到所有特征的名称：

# %%
#id , bound_id ,weight对于数据本身来说是没用的。
#输入的特征名称：
inputs_cols = ['current_coupon',
       'time_to_maturity', 'is_callable', 'reporting_delay', 'trade_size',
       'trade_type', 'curve_based_price', 'received_time_diff_last1',
       'trade_price_last1', 'trade_size_last1', 'trade_type_last1',
       'curve_based_price_last1', 'received_time_diff_last2',
       'trade_price_last2', 'trade_size_last2', 'trade_type_last2',
       'curve_based_price_last2', 'received_time_diff_last3',
       'trade_price_last3', 'trade_size_last3', 'trade_type_last3',
       'curve_based_price_last3', 'received_time_diff_last4',
       'trade_price_last4', 'trade_size_last4', 'trade_type_last4',
       'curve_based_price_last4', 'received_time_diff_last5',
       'trade_price_last5', 'trade_size_last5', 'trade_type_last5',
       'curve_based_price_last5']
#要预测的名称：
target_col = ['trade_price']

# %% [markdown]
# 根据上面的 inputs_cols 和 target_col创建训练集的样本数据和标签数据。
# 测试集合的样本数据和标签数据。

# %%
train_inputs = X_train[inputs_cols].copy()
train_target = X_train[target_col].copy()
test_inputs = X_test[inputs_cols].copy()
test_target = X_test[target_col].copy()

# %%
train_inputs

# %%
train_target

# %%
test_inputs

# %%
test_target

# %% [markdown]
# target现在是dataframe结构，而且是二维的。
# 我们转换为1维的数组，方便后面计算。

# %%
train_target1 = train_target.values.flatten()
test_target1 = test_target.values.flatten()

#解释：
#这段代码的作用是将 train_target 数据转换为一维数组，以便可以用于机器学习模型，特别是当使用 sklearn 时
#train_target 是一个 Pandas DataFrame 或 Series，其中包含了训练数据的目标值
#train_target.values 是 Pandas 的一种访问方式，用来提取 DataFrame 或 Series 中的值，并返回一个 NumPy 数组。
#flatten() 是 NumPy 数组的一种方法，它将一个多维数组转换为一维数组。
#如果 train_target.values 是一个二维数组，flatten() 方法会将其展开成一个一维数组，使得每个元素都按顺序排列。

# %%
print(train_target1)
print(type(train_target1))
#可以看到 展开之后 第一个样本的值 就在数组的第一个元素。

# %% [markdown]
# ## 2.2区分数据特征的类别，方便之后对于表示类别的特征数据进行**独热编码**

# %% [markdown]
# 根据特征的数据类别区分好了数据类型：

# %%
#这里已经全部弄好了。
#对于输入的数据的类型的分类：
numeric_cols = ['current_coupon',
       'time_to_maturity', 'reporting_delay', 'trade_size',
       'curve_based_price', 'received_time_diff_last1',
       'trade_price_last1', 'trade_size_last1',
       'curve_based_price_last1', 'received_time_diff_last2',
       'trade_price_last2', 'trade_size_last2',
       'curve_based_price_last2', 'received_time_diff_last3',
       'trade_price_last3', 'trade_size_last3',
       'curve_based_price_last3', 'received_time_diff_last4',
       'trade_price_last4', 'trade_size_last4',
       'curve_based_price_last4', 'received_time_diff_last5',
       'trade_price_last5', 'trade_size_last5',
       'curve_based_price_last5']

categorical_cols = ['is_callable', 'trade_type', 'trade_type_last1', 'trade_type_last2', 'trade_type_last3', 'trade_type_last4', 'trade_type_last5']

# %% [markdown]
# ## 2.3对数值数据进行缺失值、异常值处理

# %% [markdown]
# 在机器学习中，许多模型无法处理缺失的数值数据。插补（Imputation） 是一种填充缺失值的过程，其目的是为缺失的数据找到一个合理的替代值，使得模型可以继续处理数据。
# 
# 使用 SimpleImputer 类进行均值插补
# SimpleImputer 是 sklearn.impute 模块中的一个类，用于简单的数据插补。我们将使用这个类来实现均值插补.
# 
# 这里选择简单的均值插补!

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean').fit(train_df[numeric_cols])

#SimpleImputer(strategy='mean') 创建了一个 SimpleImputer 对象，并设置了 strategy='mean'，即用每列的均值来填充缺失值。
#.fit(train_df[numeric_cols]) 是 SimpleImputer 的 fit() 方法，它根据 train_df 中的数值型列（由 numeric_cols 指定）计算每列的均值

# %%
#检查一下异常值个数：
train_inputs[numeric_cols].isna().sum()

# %%
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])

# %%
train_inputs[numeric_cols].isna().sum()


# %% [markdown]
# 对test集合也进行表示数据的特征的缺失值处理：

# %%
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# %% [markdown]
# ## 2.4对特征进行缩放

# %% [markdown]
# 特征缩放（Feature Scaling） 是一个重要的步骤，尤其是当不同特征的值范围差异较大时。不同的数值范围会导致某些特征在模型训练过程中对损失函数的影响过大，从而使模型训练不稳定或收敛速度变慢。通过缩放特征到相同的范围（例如 $(0, 1)$ 或 $(-1, 1)$），可以确保每个特征对模型的贡献是平等的。
# 
# 我们数据集中的数值列具有不同的范围。
# 
# 让我们使用 sklearn.preprocessing 中的 MinMaxScaler 来将数值缩放到 $(0,1)$ 范围。

# %%
train_inputs[numeric_cols]

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(train_df[numeric_cols])
#调库，这里的方式，跟上面的补充缺失值 是一样的。
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
#同时 测试数据也进行了特征处理。

# %%
train_inputs[numeric_cols]

# %% [markdown]
# ## 编码分类数据

# %% [markdown]
# 由于机器学习模型只能使用数值数据进行训练，我们需要将分类数据转换为数字。常用的技术是对分类列使用 独热编码（One-hot Encoding）。
# 
# 
# 独热编码涉及为分类列中的每个唯一类别添加一个新的二进制（0/1）列。

# %%
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)

# %%
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# %% [markdown]
# train_inputs

# %%
#看一下当前数据 现在是将所有的东西 都放在一起 把离散的分类特征进行了encode。添加在了后面
train_inputs

# %%
#把原先数据里面的categorical_cols剔除掉，就得到了最终的数据。
train = train_inputs[numeric_cols + encoded_cols]
test = test_inputs[numeric_cols + encoded_cols]

# %%
train

# %%
test

# %% [markdown]
# # 3.线性回归模型预测
# 

# %% [markdown]
# 线性回归是解决回归问题时常用的技术。在线性回归模型中，目标变量被建模为输入特征的线性组合（或加权和）。模型的预测结果通过像均方根误差（RMSE）这样的损失函数进行评估。
# 
# 然而，当我们有大量输入列且存在共线性时（即某一列的值与其他列的值高度相关），线性回归的泛化能力通常较差。因为线性回归尝试完美地拟合训练数据，这可能会导致过拟合。
# 
# 为了解决这个问题，我们将使用 岭回归（Ridge Regression），它是线性回归的一种变体，采用称为 L2 正则化 的技术，通过引入另一个损失项来强迫模型更好地泛化。

# %%
from sklearn.linear_model import Ridge

model = Ridge().fit(train, train_target1)

model.fit(train, train_target1)
#得到训练数据集的预测结果：
train_preds = model.predict(train)

# %% [markdown]
# 模型现在已经训练完成，我们可以使用它来生成训练集和验证集的预测结果。我们可以通过均方根误差（RMSE）损失函数来评估模型的表现。

# %%
from sklearn.metrics import mean_squared_error

train_rmse = np.sqrt(mean_squared_error(train_target,train_preds))

print('The RMSE loss for the training set is $ {}.'.format(train_rmse))

# %%
#测试集合：
test_preds = model.predict(test)

test_rmse = np.sqrt(mean_squared_error(test_target1,test_preds))

print('The RMSE loss for the validation set is $ {}.'.format(test_rmse))

# %% [markdown]
# **特征重要性（Feature Importance）**:
# 
# 通过查看模型为不同特征分配的权重，我们可以了解哪些列对模型的预测最为重要。这是一个评估特征对模型贡献度的常见方法，特别是在使用线性回归或岭回归等模型时。模型会根据特征的相关性和重要性为每个特征分配一个权重值。

# %%
weights = model.coef_
weights_df = pd.DataFrame({
    'columns': train.columns,  # 获取训练集中的列名
    'weight': weights          # 获取模型为每个特征分配的权重
}).sort_values('weight', ascending=False)  # 按照权重降序排列
weights_df.head(10)  # 显示权重最高的前10个特征

# %% [markdown]
# # 4.随机森林模型预测

# %% [markdown]
# 一个更有效的策略是将使用略有不同的参数训练的多个决策树的结果进行组合。这种方法称为随机森林模型。
# 
# 这里的关键思想是森林中的每个决策树都会犯不同类型的错误，但通过平均，其中许多错误会相互抵消。这一想法也常被称为“群体智慧”。
# 
# 我们将使用来自`sklearn.ensemble`的`RandomForestClassifier`类。

# %% [markdown]
# 
# `n_jobs` 允许随机森林使用多个并行的工人来训练决策树，而 `random_state=42` 确保每次执行都能得到相同的结果。

# %%
from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestRegressor(n_jobs=-1, random_state=42)

rf1.fit(train, train_target1)

# %%
rf1_train_preds = rf1.predict(train)

rf1_train_rmse = np.sqrt(mean_squared_error(train_target1, rf1_train_preds))

# %%
print(rf1_train_rmse)

# %% [markdown]
# 预测结果：

# %%
rf1_test_preds = rf1.predict(test)

rf1_test_rmse = np.sqrt(mean_squared_error(test_target1, rf1_test_preds))

# %%
print(rf1_test_rmse)

# %% [markdown]
# **特征重要性**
# 
# 随机森林还会为每个特征分配一个“重要性”值，这是通过将单个树的“重要性”值进行组合得出的。

# %%
rf1_importance_df = pd.DataFrame({
    'feature': train.columns,
    'importance': rf1.feature_importances_
}).sort_values('importance', ascending=False)


rf1_importance_df.head(10)

# %%
import seaborn as sns
plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=rf1_importance_df.head(10), x='importance', y='feature');


