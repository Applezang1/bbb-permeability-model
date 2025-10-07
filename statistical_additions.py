from data_analysis import *

'''Find P-value between TPSA of BBB+ and BBB- molecules'''
tpsa_positive_array = np.array(tpsa_positive)
tpsa_negative_array = np.array(tpsa_negative)
t_stat, p_value = ttest_ind(tpsa_positive_array, tpsa_negative_array)
print(f'The P-value between TPSA of BBB+ and BBB- molecules is {p_value:.10000f}')

'''Find P-value between logP of BBB+ and BBB- molecules'''
logP_positive_array = np.array(logP_positive)
logP_negative_array = np.array(logP_negative)
t_stat, p_value = ttest_ind(logP_positive_array, logP_negative_array)
print(f'The P-value between logP of BBB+ and BBB- molecules is {p_value:.10000f}')

'''Find Confidence Interval of TPSA difference of BBB+ and BBB- molecules'''
n1 = len(tpsa_positive) # Size of the BBB+ dataset
n2 = len(tpsa_negative) # Size of the BBB- dataset
mean1 = np.mean(tpsa_positive_array)
mean2 = np.mean(tpsa_negative_array)
sd1 = np.std(tpsa_positive_array, ddof = 1) # Sample Standard Deviation of TPSA of BBB+ dataset
sd2 = np.std(tpsa_negative_array, ddof = 1) # Sample Standard Deviation of TPSA of BBB- dataset
pooled_standard_deviation = np.sqrt(((n1-1)*(sd1**2)+(n2-1)*(sd2**2))/(n1+n2-2))
standard_error = pooled_standard_deviation * np.sqrt((1/n1)+(1/n2))
s1 = np.var(tpsa_positive_array, ddof = 1) # Sample variance of TPSA of BBB+ dataset
s2 = np.var(tpsa_negative_array, ddof = 1) # Sample variance of TPSA of BBB- dataset
numerator = ((s1/n1)+ (s2/n2))**2
denominator = (((s1/n1)**2)/(n1-1))+ (((s2/n2)**2)/(n2-1))
degrees_of_freedom = numerator/denominator # Calculate degrees of freedom using the Welch-Satterthwaite formula for unequal variance
t_critical_value = stats.t.ppf(0.975, degrees_of_freedom) # Critical Value for 95% confidence 
critical_value_right = round(((mean1-mean2) + t_critical_value * np.sqrt((s1/n1)+(s2/n2))), 5) # + Critical Value
critical_value_left = round(((mean1-mean2) - t_critical_value * np.sqrt((s1/n1)+(s2/n2))), 5) # - Critical Value
print(critical_value_left)
print(critical_value_right)

'''Find Effect Size of TPSA difference of BBB+ and BBB- molecules'''
effective_size = (mean1 - mean2)/pooled_standard_deviation # Cohen's d for effective size
print(effective_size)

'''Find Confidence Interval of logP difference of BBB+ and BBB- molecules'''
n1 = len(logP_positive) # Size of the BBB+ dataset
n2 = len(logP_negative) # Size of the BBB- dataset
mean1 = np.mean(logP_positive_array)
mean2 = np.mean(logP_negative_array)
sd1 = np.std(logP_positive_array, ddof = 1) # Sample Standard Deviation of TPSA of BBB+ dataset
sd2 = np.std(logP_negative_array, ddof = 1) # Sample Standard Deviation of TPSA of BBB- dataset
pooled_standard_deviation = np.sqrt(((n1-1)*(sd1**2)+(n2-1)*(sd2**2))/(n1+n2-2))
standard_error = pooled_standard_deviation * np.sqrt((1/n1)+(1/n2))
s1 = np.var(logP_positive_array, ddof = 1) # Sample variance of TPSA of BBB+ dataset
s2 = np.var(logP_negative_array, ddof = 1) # Sample variance of TPSA of BBB- dataset
numerator = ((s1/n1)+ (s2/n2))**2
denominator = (((s1/n1)**2)/(n1-1))+ (((s2/n2)**2)/(n2-1))
degrees_of_freedom = numerator/denominator # Calculate degrees of freedom using the Welch-Satterthwaite formula for unequal variance
t_critical_value = stats.t.ppf(0.975, degrees_of_freedom) # Critical Value for 95% confidence 
critical_value_right = round(((mean1-mean2) + t_critical_value * np.sqrt((s1/n1)+(s2/n2))), 5) # + Critical Value
critical_value_left = round(((mean1-mean2) - t_critical_value * np.sqrt((s1/n1)+(s2/n2))), 5) # - Critical Value
print(critical_value_left)
print(critical_value_right)

'''Find Effect Size of logP difference of BBB+ and BBB- molecules'''
effective_size = (mean1 - mean2)/pooled_standard_deviation # Cohen's d for effective size
print(effective_size)
