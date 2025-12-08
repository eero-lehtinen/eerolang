def fibonacci(n):
    if n <= 1:
        return n
    sum = fibonacci(n - 1) + fibonacci(n - 2)
    print(sum)
    return sum


print(fibonacci(4))
