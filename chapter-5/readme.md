# Array summation with np.sum() and Broadcasting in Python

## Introduction

In Python, especially when using the NumPy library, `np.sum()` is a commonly used function to compute the sum of array elements over a specified axis. This README explains how `np.sum()` works, including its behavior with multi-dimensional arrays and the concept of broadcasting that often accompanies array operations in NumPy.

## np.sum() Function

The `np.sum()` function in NumPy computes the sum of all elements in an array or along a specified axis. It returns the sum of array elements or sums across the specified dimensions.

### Syntax

```python
np.sum(a, axis=None, dtype=None, out=None, keepdims=False)
```

### Example 1
```python
import numpy as np

a = np.array([1, 2, 3, 4])
total_sum = np.sum(a)
print(total_sum)  # Output: 10
```

### Example 2
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

# Sum along the rows (axis=1)
row_sum = np.sum(a, axis=1)
print(row_sum)  # Output: [ 6 15]

# Sum along the columns (axis=0)
column_sum = np.sum(a, axis=0)
print(column_sum)  # Output: [5 7 9]
```

# NumPy Broadcasting

## Introduction

Broadcasting in NumPy is a powerful mechanism that allows operations on arrays of different shapes. It eliminates the need for manually replicating smaller arrays to match the size of larger ones, making array operations more efficient and less memory-intensive. This README provides an overview of broadcasting in NumPy, along with illustrative examples.

## Broadcasting Rules

Broadcasting works by following these rules when performing operations on arrays with different shapes:

1. **Comparing Shapes**: NumPy compares the shapes of the two arrays element by element, starting from the rightmost dimension.
2. **Compatible Dimensions**: Two dimensions are compatible for broadcasting if:
    - They are equal, or
    - One of them is 1 (the smaller array can be "stretched" to match the larger one).
3. **Result Shape**: The resulting shape is derived from the maximum dimensions of the two arrays being operated on.

### Example 1: Scalar and 1D Array

When a scalar value is added to a 1D array, the scalar is broadcasted to match the shape of the array.

```python
import numpy as np

a = np.array([1, 2, 3])
b = 5

# Broadcasting: b is treated as an array [5, 5, 5]
result = a + b
print(result)  # Output: [6 7 8]
```
### Example 2: A 1D array can be added to a 2D array. The 1D array is broadcasted across each row of the 2D array.
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])

# Broadcasting: b is treated as [[1, 2, 3], [1, 2, 3]]
result = a + b
print(result)
# Output:
# [[2 4 6]
#  [5 7 9]]

```

### Example 3: Broadcasting with Different 2D Arrays
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1], [2]])

# Broadcasting: b is treated as [[1, 1, 1], [2, 2, 2]]
result = a + b
print(result)
# Output:
# [[2 3 4]
#  [6 7 8]]

```

### Example 4: Incompatible Shapes
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# This will raise a ValueError because the shapes are not compatible
try:
    result = a + b
except ValueError as e:
    print(f"Error: {e}")

```



