import sympy as sp

# Define symbols
t, f = sp.symbols("t f")

# Define the signal function (example: a simple Gaussian pulse)
signal = sp.exp(-(t**2))

# Compute the Fourier Transform of the signal
fourier_transform = sp.fourier_transform(signal, t, f)

# Take the log magnitude of the spectrum
log_magnitude_spectrum = sp.log(sp.Abs(fourier_transform))

# Compute the Inverse Fourier Transform manually using the integral formula
cepstrum_integral = sp.inverse_fourier_transform(log_magnitude_spectrum, f, t)

# Take the square of the absolute value of the cepstrum
cepstrum_square_abs = sp.Abs(cepstrum_integral) ** 2

# Display the resulting cepstrum expression
print("Cepstrum:", cepstrum_square_abs)
