# example.py


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Some code to test
print(add(2, 3))
print(subtract(5, 2))
print(multiply(3, 4))
print(divide(6, 2))
