import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate

clfs = {
    'ONE': "one",
    'TWO': "two",
    'THREE': "three",
    'FOUR': "four",
    'FIVE': "five"
}

results1 = np.load("results/statistical/cross_valid_1.npy")
# print(results1)
results2 = np.load("results/statistical/cross_valid_2.npy")
# print(results2)
results3 = np.load("results/statistical/cross_valid_3.npy")
# print(results3)
results4 = np.load("results/statistical/cross_valid_4.npy")
# print(results4)
results5 = np.load("results/statistical/cross_valid_5.npy")
# print(results5)

results_arr = np.array((results1, results2, results3, results4, results5))

print(results_arr)

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(results_arr[i], results_arr[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

headers = ["1", "2", "3", "4", "5"]
names_column = np.array([["1"], ["2"], ["3"],["4"], ["5"]])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)

mean = np.mean(results_arr, axis=1)
std = np.std(results_arr, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))