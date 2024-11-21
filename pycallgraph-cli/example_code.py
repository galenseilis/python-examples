# example_code.py


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b


def main():
    x = add(5, 3)
    y = subtract(10, 4)
    z = multiply(x, y)
    result = divide(z, 2)
    print("Result:", result)


if __name__ == "__main__":
    main()
