def AssignGroup(x, bin):
    N = len(bin)
    if x<=min(bin):
        return min(bin)
    elif x>=max(bin):
        return max(bin)
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1] 
			

def Chi2(df, total_col, bad_col, good_col,overallRate_bad, overallRate_good):
    df2 = df.copy()
    df2['expected_bad'] = df[total_col].apply(lambda x: x*overallRate_bad)
    df2['expected_good'] = df[total_col].apply(lambda x: x*overallRate_good)
    combined = zip(df2['expected_bad'], df2[bad_col], df2['expected_good'], df2[good_col])
    chi1 = sum([(i[0]-i[1])**2/i[0] for i in combined])
    chi2 = sum([(i[2]-i[3])**2/i[2] for i in combined])
    chi3 = chi1 + chi2
    return chi3
	
def AssignBin(x, cutOffPoints,special_attribute=[]):
    numBin = len(cutOffPoints) + 1 
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    elif x<=cutOffPoints[0]:
        return 'Bin 1, (-1,{}]'.format(cutOffPoints[0])
    elif x > cutOffPoints[-1]:
        return 'Bin {0}, ({1},inf)'.format(numBin,cutOffPoints[-1])
    else:
        for i in range(0,numBin-2):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {0}, ({1},{2}]'.format(i+2, cutOffPoints[i], cutOffPoints[i+1])
				
def TransferWOE(df_train,df,df2,special_attribute = []):
    for col in allFeatures3:
        if col in numerical_var:
            special_attribute = [-1]
        if col in numerical_var2:
            special_attribute = []
        cutOffPoints = CUTOFF[col]
        col1 = str(col) + '_Bin'
        df_train[col1] = df_train[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        total_train = df_train.groupby([col1])['y'].count()
        total_train = pd.DataFrame({'total': total_train})
        bad_train = df_train.groupby([col1])['y'].sum()
        bad_train = pd.DataFrame({'bad': bad_train})
        regroup = total_train.merge(bad_train, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        regroup['good'] = regroup['total'] - regroup['bad']
        G = N - B
        regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
        regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
        regroup['bad_pcnt_bin'] = regroup['bad'] * 1.0 /regroup['total']
        regroup['total_pcnt'] = regroup['total'].map(lambda x : x * 1.0 / N)
        regroup['WOE'] = regroup.apply(lambda x: np.log(x.bad_pcnt*1.0/x.good_pcnt),axis = 1)
        
        df[col1] = df[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        WOE = pd.merge(df[[col,col1]], regroup[['WOE', col1]], how='left', left_on=col1, right_on=col1)
        col2 = str(col) + '_WOE'
        df2[col2] = WOE['WOE']
    return df2
	
	def CalculateIV(df,special_attribute = []):
    for col in allFeatures3:
        if col in numerical_var:
            special_attribute = [-1]
        elif col in numerical_var2:
            special_attribute = []
        df2 = df.loc[~df[col].isin([-1])]
        colLevels = sorted(list(set(df[col])))   
    #     N_distinct = len(list(set(df2[col])))  
    #     if N_distinct <= 1000:
    #         split_x = [i for i in colLevels]
    #     else:
    #          等宽分箱
    #         min_col = min(list(set(df[col])))
    #         max_col = max(list(set(df[col])))
    #         split_x = [i for i in np.arange(min_col, max_col, (max_col - min_col) / 1000.0)]
    #      等频分箱
        split_x = list(pd.unique(df[col].quantile(np.linspace(0,1,21))))
        df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        total = df2.groupby(['temp'])['y'].count()
        total = pd.DataFrame({'total': total})
        bad = df2.groupby(['temp'])['y'].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        regroup['good'] = regroup['total'] - regroup['bad']
        overallRate_bad = B * 1.0 / N
        overallRate_good = 1 - overallRate_bad
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        #Indicate the number of bins 
        if len(special_attribute)>=1:
            split_intervals = 6 - len(special_attribute)
        else:
            split_intervals = 5 
        while len(groupIntervals) > split_intervals:  
                    chisqList = []
                    for interval in groupIntervals:               
                        df2b = regroup.loc[regroup['temp'].isin(interval)]
                        chisq = Chi2(df2b, 'total', 'bad', 'good', overallRate_bad, overallRate_good)
                        chisqList.append(chisq)
                    min_position = chisqList.index(min(chisqList))
                    if min_position == 0:
                        combinedPosition = 1
                    elif min_position == groupNum - 1:
                        combinedPosition = min_position - 1
                    else:
                        if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                            combinedPosition = min_position - 1
                        else:
                            combinedPosition = min_position + 1
                    groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
                    groupIntervals.remove(groupIntervals[combinedPosition])
                    groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints= [max(i) for i in groupIntervals[:-1]]
        CUTOFF[col] = cutOffPoints
        col1 = str(col) + '_Bin'      
        df[col1] = df[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        total = df.groupby([col1])['y'].count()
        total = pd.DataFrame({'total': total})
        bad = df.groupby([col1])['y'].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        regroup['good'] = regroup['total'] - regroup['bad']
        G = N - B
        regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
        regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
        regroup['bad_pcnt_bin'] = regroup['bad'] * 1.0 /regroup['total']
        regroup['total_pcnt'] = regroup['total'].map(lambda x : x * 1.0 / N)   
        regroup['WOE'] = regroup.apply(lambda x: np.log(x.bad_pcnt*1.0/x.good_pcnt),axis = 1)
        regroup['IV'] = regroup.apply(lambda x: (x.bad_pcnt-x.good_pcnt)*np.log(x.bad_pcnt*1.0/x.good_pcnt),axis = 1)
        regroup['IV'][regroup['IV'] == float('inf')] = 0
        IV = regroup['IV'].sum()
        if regroup['bad_pcnt'].astype(np.float64).values.max() > 0.80:
            print('{0} #bad_pcnt>80%# IV is: {1}'.format(col,IV))
        else:
            print('{0} IV is: {1}'.format(col,IV))
        var_IV = pd.DataFrame({'var':col,'IV':IV},index=[0])
        var_IV.to_csv('',index = True, header = True,mode = 'a')
        regroup.to_csv('',index = False, header = True,mode = 'a')
		
# calculate the KS value
total = df_train.groupby(['score'])['y'].count()
bad = df_train.groupby(['score'])['y'].sum()
all_train = pd.DataFrame({'total':total, 'bad':bad})
all_train['good'] = all_train['total'] - all_train['bad']
all_train['score'] = all_train.index
all_train = all_train.sort_values(by='score',ascending=True)
all_train.index = range(len(all_train))
all_train['badCumRate'] = all_train['bad'].cumsum() / all_train['bad'].sum()
all_train['goodCumRate'] = all_train['good'].cumsum() / all_train['good'].sum()
all_train['totalPcnt'] = all_train['total'].cumsum() / all_train['total'].sum()
KS = all_train.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
all_train['KS'] = all_train['badCumRate'] - all_train['goodCumRate']
print(max(KS))

# plot the KS curve
from scipy.interpolate import spline  

all_train_quantile = all_train.quantile(np.linspace(0,1,51))
all_train_quantile['KS'] = all_train_quantile['badCumRate'] - all_train_quantile['goodCumRate']

x = all_train_quantile.index
y1 = all_train_quantile['badCumRate']
y2 = all_train_quantile['goodCumRate']
y3 = all_train_quantile['KS']

plt.plot(x,y1,label = 'badCumRate',linewidth = '3')  
plt.plot(x,y2,label = 'goodCumRate',linewidth = '3')  
plt.plot(x,y3,label = 'KS',linewidth = '3')  
plt.legend()
plt.grid(True,linestyle = '--')
plt.title('Train KS: %(KS).2f' % {'KS':max(KS)})
