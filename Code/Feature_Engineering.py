#this capstone project is written in python IDLE 3.5.3
#in this section, I am creating a function that would create the necessary dataset to plug into different models
#I'm also going to test out whether datasets that went through PCA will perform better or not, so in the data_wrangle function, I'm also going to output PCA datasets

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import visuals as vs

color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

def data_wrangle(): #this is the main method that we call on to perform feature engineering
    #import data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train.drop('Id', axis = 1, inplace = True)
    test.drop('Id', axis = 1, inplace = True)

    #remove outliers
    outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)]
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    if False: #replot outliers 'GrLivArea'
        fig, ax = plt.subplots()
        ax.scatter(train['GrLivArea'], train['SalePrice'])
        ax.scatter(outliers['GrLivArea'], outliers['SalePrice'], color = 'r')
        plt.ylabel('SalePrice', fontsize = 13)
        plt.xlabel('GrLivArea', fontsize = 13)
        plt.title('GrLivArea vs. SalePrice - outliers')
        plt.tight_layout()
        plt.savefig('GrLivArea Outlier.png')
        plt.show()

    if False: #Target 'SalePrice', trying to correct for skewness
        lam = 0.15
        fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15,6))
        sns.distplot(train['SalePrice'], fit = norm, ax = ax1)
        sns.distplot(boxcox1p(train['SalePrice'], lam), fit = norm, ax = ax2)
        sns.distplot(np.log1p(train['SalePrice']), fit = norm, ax = ax3)
        # Get the fitted parameters used by the function
        (mu1, sigma1) = norm.fit(train['SalePrice'])
        (mu2, sigma2) = norm.fit(boxcox1p(train['SalePrice'], lam))
        (mu3, sigma3) = norm.fit(np.log1p(train['SalePrice']))
        ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                    'Skewness: {:.2f}'.format(skew(train['SalePrice']))],
                    loc = 'best')
        ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                    'Skewness: {:.2f}'.format(skew(boxcox1p(train['SalePrice'], lam)))],
                    loc = 'best')
        ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma3),
                    'Skewness: {:.2f}'.format(skew(np.log1p(train['SalePrice'])))],
                    loc = 'best')
        ax1.set_ylabel('Frequency')
        ax1.set_title('SalePrice Distribution')
        ax2.set_title('SalePrice Box-Cox Transformed')
        ax3.set_title('SalePrice Log Transformed')
        plt.tight_layout()
        plt.savefig('SalePrice Distribution.png')
        plt.show()

        #Get also the QQ-plot
        if True:
            fig = plt.subplots(figsize = (15,6))
            ax1 = plt.subplot(131)
            res = stats.probplot(train['SalePrice'], plot = plt)
            ax1.set_title('SalePrice Probability Plot')
            
            ax2 = plt.subplot(132)
            res = stats.probplot(boxcox1p(train['SalePrice'], lam), plot = plt)
            ax2.set_title('SalePrice Box-Cox Transformed Probability Plot')
            
            ax3 = plt.subplot(133)
            res = stats.probplot(np.log1p(train['SalePrice']), plot = plt)
            ax3.set_title('SalePrice Log Transformed Probability Plot')
            
            plt.tight_layout()
            plt.savefig('SalePrice Probability Plot.png')
            plt.show()

    train['SalePrice'] = np.log1p(train['SalePrice'])
    test['SalePrice'] = np.nan

    #get some stats from the data as a whole
    ntrain = train.shape[0]
    ntest = test.shape[0]
    ytrain = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop = True)

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

    if False: #missing data visualization
        f, ax = plt.subplots(figsize=(10, 7))
        plt.xticks(rotation = '90')
        sns.barplot(x = all_data_na.index, y = all_data_na)
        plt.xlabel('Features', fontsize = 15)
        plt.ylabel('Percent of missing values', fontsize = 15)
        plt.title('Percent NaNs by Features', fontsize = 15)
        plt.tight_layout()
        plt.savefig('Percent NaNs by Features.png')
        plt.show()

    if False: #correlation X vs. ylog
        corrmat = train.corr()
        plt.subplots(figsize = (12, 9))
        g = sns.heatmap(corrmat, vmax = 0.9, square = True)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
        plt.title('Correlation Matrix/Heatmap Numerical Features vs. ylog')
        plt.tight_layout()
        plt.savefig('Numerical Features vs. ylog heatmap.png')
        plt.show()

    # look at missind data:
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})

    '''
                  Missing Ratio
    PoolQC            99.691464
    MiscFeature       96.400411
    Alley             93.212204
    Fence             80.425094
    SalePrice         50.017141
    FireplaceQu       48.680151
    LotFrontage       16.660953
    GarageYrBlt        5.450806
    GarageFinish       5.450806
    GarageQual         5.450806
    GarageCond         5.450806
    GarageType         5.382242
    BsmtCond           2.811107
    BsmtExposure       2.811107
    BsmtQual           2.776826
    BsmtFinType2       2.742544
    BsmtFinType1       2.708262
    MasVnrType         0.822763
    MasVnrArea         0.788481
    MSZoning           0.137127
    BsmtFullBath       0.068564
    BsmtHalfBath       0.068564
    Functional         0.068564
    Utilities          0.068564
    BsmtFinSF2         0.034282
    BsmtUnfSF          0.034282
    BsmtFinSF1         0.034282
    TotalBsmtSF        0.034282
    SaleType           0.034282
    KitchenQual        0.034282
    Exterior2nd        0.034282
    Exterior1st        0.034282
    GarageCars         0.034282
    GarageArea         0.034282
    Electrical         0.034282
    '''

    # In this section, we are browsing over the features to make sure everything looks okay

    # those that are NaN but actually should be None:
    None_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
                 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
    if False:
        for col in None_cols:
            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,7), sharey = True)
            sns.boxplot(data = all_data[[col, 'SalePrice']], x = col, y = 'SalePrice', ax = ax1)
            ax1.set_title('{} Distribution: {:.3f}% is NaN'.format(col, missing_data.ix[col, 'Missing Ratio']))
            all_data[[col]] = all_data[col].fillna('None')
            sns.boxplot(data = all_data[[col, 'SalePrice']], x = col, y = 'SalePrice', ax = ax2)
            ax2.set_title(col + 'NaN Imputed with None')
            plt.tight_layout()
            plt.savefig(col + ' Distribution (Replace NaNs).png')
            plt.show()

    # PoolQC(99.691464): data description says NA means 'No Pool'.
    all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
    # MiscFeature(96.400411): data description says NA means 'no misc feature'
    all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
    # Alley(93.212204): data description says NA means 'no alley access'
    all_data['Alley'] = all_data['Alley'].fillna('None')
    # Fence(80.425094): data description says NA means 'no fence'
    all_data['Fence'] = all_data['Fence'].fillna('None')
    # FireplaceQu(48.680151): data description says NA means 'no fireplace'
    all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
    # GarageType(5.382242), GarageFinish(5.450806), GarageQual(5.450806) and GarageCond (5.450806): Replacing missing data with None
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    # BsmtQual(2.776826), BsmtCond(2.811107), BsmtExposure(2.811107), BsmtFinType1(2.708262) and BsmtFinType2(2.742544): For all these categorical basement-related features, NaN means that there is no basement.
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')    
    # MasVnrType(0.822763): NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
    all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

    #those that are NaN but should be zero
    Zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    if False:
        for col in Zero_cols:
            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,7), sharey = True)
            ax1.scatter(all_data[col], all_data['SalePrice'])
            ax1.set_title('{} Distribution: {:.3f}% is NaN'.format(col, missing_data.ix[col, 'Missing Ratio']))
            ax1.set_xlabel(col)
            ax1.set_ylabel('SalePrice')
            all_data[[col]] = all_data[col].fillna(0)
            ax2.scatter(all_data[col], all_data['SalePrice'])
            ax2.set_title(col + 'NaN Imputed with 0')
            ax2.set_xlabel(col)
            plt.tight_layout()
            plt.savefig(col + ' vs. SalePrice Scatter (Replace NaNs).png')
            plt.show()

    # GarageYrBlt(5.450806), GarageArea(0.034282) and GarageCars(0.034282): Replacing missing data with 0 (Since No garage = no cars in such garage.)
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    # BsmtFinSF1(0.034282), BsmtFinSF2(0.034282), BsmtUnfSF(0.034282), TotalBsmtSF(0.034282), BsmtFullBath(0.068564) and BsmtHalfBath (): missing values are likely zero for having no basement
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    # MasVnrArea(0.788481): NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

    #those that are NaN but should be the mode (most occured, sorta like average for categorical variables)
    Mode_cols = ['MSZoning', 'Electrical', 'KitchenQual',
                 'Exterior1st', 'Exterior2nd', 'SaleType']
    if False:
        for col in Mode_cols:
            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,7), sharey = True)
            sns.boxplot(data = all_data[[col, 'SalePrice']], x = col, y = 'SalePrice', ax = ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 30)
            ax1.set_title('{} Distribution: {:.3f}% is NaN'.format(col, missing_data.ix[col, 'Missing Ratio']))
            all_data[[col]] = all_data[col].fillna(all_data[col].mode()[0])
            sns.boxplot(data = all_data[[col, 'SalePrice']], x = col, y = 'SalePrice', ax = ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 30)
            ax2.set_title(col + 'NaN Imputed with Mode')
            plt.tight_layout()
            plt.savefig(col + ' Distribution (Replace NaNs).png')
            plt.show()

    # MSZoning(0.137127) (The general zoning classification): 'RL' is by far the most common value. So we can fill in missing values with 'RL'
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    # Electrical(0.034282): It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    # KitchenQual(0.034282): Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    # Exterior1st(0.034282) and Exterior2nd(0.034282): Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    # SaleType(0.034282): Fill in again with most frequent which is 'WD'
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    # those that are NaN should be Typ
    if False:
        col = 'Functional'
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,7), sharey = True)
        sns.boxplot(data = all_data[[col, 'SalePrice']], x = col, y = 'SalePrice', ax = ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 30)
        ax1.set_title('{} Distribution: {:.3f}% is NaN'.format(col, missing_data.ix[col, 'Missing Ratio']))
        all_data[[col]] = all_data[col].fillna('Typ')
        sns.boxplot(data = all_data[[col, 'SalePrice']], x = col, y = 'SalePrice', ax = ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 30)
        ax2.set_title(col + 'NaN Imputed with Typ')
        plt.tight_layout()
        plt.savefig(col + ' Distribution (Replace NaNs).png')
        plt.show()

    # Functional(0.068564): data description says NA means typical
    all_data['Functional'] = all_data['Functional'].fillna('Typ')

    # those that are NaN should be the Median value
    if False:
        col = 'LotFrontage'
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,7), sharey = True)
        ax1.scatter(all_data[col], all_data['SalePrice'])
        #ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 30)
        ax1.set_title('{} Distribution: {:.3f}% is NaN'.format(col, missing_data.ix[col, 'Missing Ratio']))
        ax1.set_xlabel(col)
        ax1.set_ylabel('SalePrice')
        all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        ax2.scatter(all_data[col], all_data['SalePrice'])
        #ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 30)
        ax2.set_title(col + 'NaN Imputed with Median')
        ax2.set_xlabel(col)
        plt.tight_layout()
        plt.savefig(col + ' Distribution.png')
        plt.show()
        
    # LotFrontage(16.660953): Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    #some columns needs to be removed
    if False:
        all_data['Utilities'] = all_data['Utilities'].fillna('None')
        sns.barplot(x = all_data.groupby('Utilities')['Utilities'].count().index, y = all_data.groupby('Utilities')['Utilities'].count().values)
        plt.title('Utilities Distribution: {:.3f}% is NaN'.format(missing_data.ix['Utilities', 'Missing Ratio']))
        plt.savefig('Utilities Distribution (Removed Feature).png')
        plt.show()

    # Utilities(0.068564): For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
    all_data = all_data.drop(['Utilities'], axis = 1)


    # Transforming some numerical variables that are really categorical
    tans_cols = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
    if False:
        for col in tans_cols:
            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,7), sharey = True)
            ax1.scatter(all_data[col], all_data['SalePrice'])
            ax1.set_xlabel(col)
            ax1.set_ylabel('SalePrice')
            #ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 30)
            ax1.set_title(col + ' as Numerical Feature')
            all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)    
            sns.boxplot(all_data[col], all_data['SalePrice'])
            #ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 30)
            ax2.set_title(col + ' as Categorical Feature')
            plt.tight_layout()
            plt.savefig(col + ' Distributions (Num to Cat).png')
            plt.show()

        
    #MSSubClass = The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    #Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    #Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    #These are the categorical features that may have rankings correlated with SalePrice, try LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')

    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values)) 
        all_data[c] = lbl.transform(list(all_data[c].values))

    # Adding total sqfootage feature 
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    if False:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize = (16,7), sharey = True)
        ax1.scatter(all_data['TotalBsmtSF'], all_data['SalePrice'])
        ax1.set_title('TotalBsmtSF vs. SalePrice, corr: {:.3f}%'.format(all_data[['TotalBsmtSF', 'SalePrice']].corr().as_matrix()[0,1] * 100))
        ax2.scatter(all_data['1stFlrSF'], all_data['SalePrice'])
        ax2.set_title('1stFlrSF vs. SalePrice, corr: {:.3f}%'.format(all_data[['1stFlrSF', 'SalePrice']].corr().as_matrix()[0,1] * 100))
        ax3.scatter(all_data['2ndFlrSF'], all_data['SalePrice'])
        ax3.set_title('2ndFlrSF vs. SalePrice, corr: {:.3f}%'.format(all_data[['2ndFlrSF', 'SalePrice']].corr().as_matrix()[0,1] * 100))
        ax4.scatter(all_data['TotalSF'], all_data['SalePrice'])
        ax4.set_title('TotalSF vs. SalePrice, corr: {:.3f}%'.format(all_data[['TotalSF', 'SalePrice']].corr().as_matrix()[0,1] * 100))
        plt.tight_layout()
        plt.savefig('TotalSF Distribution (New Feature).png')
        plt.show()


    #Let's now look at numerical features
    #Skewed features

    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})

    #we can try either log transformation or box-cox transformation to see if we can fix the skew
    skewness = skewness[abs(skewness) > 0.75]

    skewed_features = skewness.index
    lam = 0.15

    if False: #trying to correct for skewness
        for feat in skewed_features:
            print(feat)
            if feat != 'SalePrice':
                fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (12,6))
                sns.distplot(all_data[feat], fit = norm, ax = ax1)
                sns.distplot(boxcox1p(all_data[feat], lam), fit = norm, ax = ax2)
                sns.distplot(np.log1p(all_data[feat]), fit = norm, ax = ax3)
                # Get the fitted parameters used by the function
                (mu1, sigma1) = norm.fit(all_data[feat])
                (mu2, sigma2) = norm.fit(boxcox1p(all_data[feat], lam))
                (mu3, sigma3) = norm.fit(np.log1p(all_data[feat]))
                ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                            'Skewness: {:.2f}'.format(skew(all_data[feat]))],
                            loc = 'best')
                ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                            'Skewness: {:.2f}'.format(skew(boxcox1p(all_data[feat], lam)))],
                            loc = 'best')
                ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma3),
                            'Skewness: {:.2f}'.format(skew(np.log1p(all_data[feat])))],
                            loc = 'best')
                ax1.set_ylabel('Frequency')
                ax1.set_title(feat + ' Distribution')
                ax2.set_title(feat + ' Box-Cox Transformed')
                ax3.set_title(feat + ' Log Transformed')
                plt.tight_layout()
                plt.savefig(feat + ' Distribution.png')
                #plt.show()

                #Get also the QQ-plot
                if True:
                    fig = plt.subplots(figsize = (12,6))
                    ax1 = plt.subplot(131)
                    res = stats.probplot(all_data[feat], plot = plt)
                    ax1.set_title(feat + ' Probability Plot')
                    
                    ax2 = plt.subplot(132)
                    res = stats.probplot(boxcox1p(all_data[feat], lam), plot = plt)
                    ax2.set_title(feat + ' Box-Cox Transformed Probability Plot')
                    
                    ax3 = plt.subplot(133)
                    res = stats.probplot(np.log1p(all_data[feat]), plot = plt)
                    ax3.set_title(feat + ' Log Transformed Probability Plot')
                    
                    plt.tight_layout()
                    plt.savefig(feat + ' Probability Plot.png')
                    #plt.show()
        
    #these are the features that need log transformation
    log_trans = ['MiscVal', 'PoolArea', 'LowQualFinSF', '3SsnPorch', 'LandSlope',
                 'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
                 'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
                 '1stFlrSF', 'GrLivArea', '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces',
                 'HalfBath', 'BsmtFullBath', 'HeatingQC']
    #these are the features that need box-cox transformation
    boxcox_trans = ['LotArea', 'LotFrontage', 'TotalSF', 'BsmtFinSF1']

    for feat in boxcox_trans:
        all_data[feat] = boxcox1p(all_data[feat], lam)
        
    for feat in log_trans:
        all_data[feat] = np.log1p(all_data[feat])

    train = all_data[:ntrain]
    test = all_data[ntrain:]

    if False: #correlation transformed X vs. ylog
        corrmat = train.corr()
        plt.subplots(figsize = (12, 9))
        g = sns.heatmap(corrmat, vmax = 0.9, square = True)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
        plt.title('Correlation Matrix/Heatmap Numerical Features vs. ylog')
        plt.tight_layout()
        plt.savefig('Transformed Numerical Features vs. ylog heatmap.png')
        plt.show()

    all_data = pd.get_dummies(all_data)
    #print(all_data.shape)
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    y = train['SalePrice']
    train.drop('SalePrice', axis = 1, inplace = True)
    test.drop('SalePrice', axis = 1, inplace = True)

    #let's also try PCA
    pca = PCA(n_components = 7)
    trainPCA = pd.DataFrame(pca.fit_transform(train))
    testPCA = pd.DataFrame(pca.transform(test))
    
    if False:
        pca_results = vs.pca_results(train, pca)
        plt.savefig('PCA.png')
        plt.show() # saved to:
        ys = pca.explained_variance_ratio_
        xs = np.arange(1, len(ys)+1)
        plt.plot(xs, np.cumsum(ys), '-o')
        for label, x, y in zip(np.cumsum(ys), xs, np.cumsum(ys)):
            plt.annotate('{:.2f}%'.format(label * 100),
                xy=(x, y), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
        plt.ylabel('Cumulative Explained Variance')
        plt.xlabel('Dimensions')
        plt.title('PCA - Total Explained Variance by # fo Dimensions')
        plt.tight_layout()
        plt.savefig('PCA Cumsum.png')
        plt.show()
    
        temp = pd.DataFrame.copy(trainPCA)
        temp.columns = ['Dimension ' + str(i) for i in range(1,8)]
        temp['SalePrice'] = ytrain

        g = sns.heatmap(temp.corr(), annot = True, annot_kws={'size': 8})
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
        plt.title('PCA Correlation Matrix/Heatmap')
        plt.tight_layout()
        plt.savefig('PCA heatmap.png')
        plt.show() # saved to:
        
    return train, test, y, trainPCA, testPCA

if __name__ == '__main__':
    train, test, y, trainPCA, testPCA = data_wrangle()
    print(train.shape, type(train))
    print(test.shape, type(test))
    print(y.shape, type(y))
    print(trainPCA.shape, type(trainPCA))
    print(testPCA.shape, type(testPCA))
