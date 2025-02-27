import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    middle = [pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    return arr


# Prepare to collect durations
results = {
    "bubble_sort": [],
    "insertion_sort": [],
    "merge_sort": [],
    "quick_sort": []
}

# Run experiments
for _ in range(100):
    random_list = np.random.randint(1, 100000, size=1000).tolist()
    start_time = time.time()
    bubble_sort(random_list.copy())
    duration = time.time() - start_time
    results['bubble_sort'].append(duration)

    start_time = time.time()
    insertion_sort(random_list.copy())
    duration = time.time() - start_time
    results['insertion_sort'].append(duration)

    start_time = time.time()
    merge_sort(random_list.copy())
    duration = time.time() - start_time
    results['merge_sort'].append(duration)

    start_time = time.time()
    quick_sort(random_list.copy())
    duration = time.time() - start_time
    results['quick_sort'].append(duration)

results_df = pd.DataFrame(results)
plt.figure(figsize=(20, 15))
sns.boxplot(data=results_df)
plt.ylabel("duration")
plt.grid()
plt.show()

mean_values = results_df.mean()
median_values = results_df.median()
standard_deviation_values = results_df.std()

statistics = pd.DataFrame({
    'mean': mean_values,
    'median': median_values,
    'standard_deviation': standard_deviation_values
})

print(statistics)
print("=" * 100)
skewed = results_df.skew()
print("skewed:\n", skewed)
print("=" * 100)
kurtosis = results_df.kurtosis()
print("Kurtosis:\n", kurtosis)
print("=" * 100)
min_standard_deviation = statistics['standard_deviation'].min()
max_standard_deviation = statistics['standard_deviation'].max()
stable = statistics[statistics['standard_deviation'] == min_standard_deviation].index[0]
volatile = statistics[statistics['standard_deviation'] == max_standard_deviation].index[0]
print(f"stable algorithm: {stable}")
print(f"volatile algorithm: {volatile}")
