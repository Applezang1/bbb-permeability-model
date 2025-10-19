from data_analysis import *
import statistics

'''Find P-value between TPSA of BBB+ and BBB- molecules'''
tpsa_positive_array = np.array(tpsa_positive)
tpsa_negative_array = np.array(tpsa_negative)
t_stat, p_value = ttest_ind(tpsa_positive_array, tpsa_negative_array)
print(f'The P-value between TPSA of BBB+ and BBB- molecules is {p_value:.10f}')

'''Find P-value between logP of BBB+ and BBB- molecules'''
logP_positive_array = np.array(logP_positive)
logP_negative_array = np.array(logP_negative)
t_stat, p_value = ttest_ind(logP_positive_array, logP_negative_array)
print(f'The P-value between logP of BBB+ and BBB- molecules is {p_value:.10f}')

'''Find the Median of TPSA for BBB+ and BBB- molecules'''
median_tpsa_positive = statistics.median(tpsa_positive_array)
median_tpsa_negative = statistics.median(tpsa_negative_array)
print(f'Median TPSA BBB+: {median_tpsa_positive}')
print(f'Median TPSA BBB-: {median_tpsa_negative}')

'''Find the Median of logP for BBB+ and BBB- molecules'''
median_logP_positive = statistics.median(logP_positive_array)
median_logP_negative = statistics.median(logP_negative_array)
print(f'Median logP BBB+: {median_logP_positive}')
print(f'Median logP BBB-: {median_logP_negative}') 

'''Find the Interquartile Range of TPSA for BBB+ and BBB- molecules'''
Q1_tpsa_positive = np.percentile(tpsa_positive_array, 25)
Q3_tpsa_positive = np.percentile(tpsa_positive_array, 75)
print(f"TPSA BBB+ Q1: {Q1_tpsa_positive}")
print(f"TPSA BBB+ Q3: {Q3_tpsa_positive}")

Q1_tpsa_negative = np.percentile(tpsa_negative_array, 25)
Q3_tpsa_negative = np.percentile(tpsa_negative_array, 75)
print(f"TPSA BBB- Q1: {Q1_tpsa_negative}")
print(f"TPSA BBB- Q3: {Q3_tpsa_negative}")

'''Find the Interquartile Range of logP for BBB+ and BBB- molecules'''
Q1_logP_positive = np.percentile(logP_positive_array, 25)
Q3_logP_positive = np.percentile(logP_positive_array, 75)
print(f"logP BBB+ Q1: {Q1_logP_positive}")
print(f"logP BBB+ Q3: {Q3_logP_positive}")

Q1_logP_negative = np.percentile(logP_negative_array, 25)
Q3_logP_negative = np.percentile(logP_negative_array, 75)
print(f"logP BBB- Q1: {Q1_logP_negative}")
print(f"logP BBB- Q3: {Q3_logP_negative}")

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
print(f'We can state that the TPSA difference of BBB+ and BBB- lies between {critical_value_left} and {critical_value_right} with 95% confidence.')

'''Find Effect Size of TPSA difference of BBB+ and BBB- molecules'''
effective_size = (mean1 - mean2)/pooled_standard_deviation # Cohen's d for effective size
print(f'The effective size for the difference between TPSA of BBB+ and BBB- is {effective_size}.')

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
print(f'We can state that the logP difference of BBB+ and BBB- lies between {critical_value_left} and {critical_value_right} with 95% confidence.')

'''Find Effect Size of logP difference of BBB+ and BBB- molecules'''
effective_size = (mean1 - mean2)/pooled_standard_deviation # Cohen's d for effective size
print(f'The effective size for the difference between logP of BBB+ and BBB- is {effective_size}.')
