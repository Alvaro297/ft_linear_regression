import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def main():
	df = pd.read_csv("./data.csv")
	x = df['km']
	y = df['price']

	x_norm = (x - x.min()) / (x.max() - x.min())
	y_norm = (y - y.min()) / (y.max() - y.min())

	a = 0.0
	b = 0.0
	lr = 0.1
	
	for _ in range(1000):
		y_pred = a + b * x_norm
		error = y_pred - y_norm
		a -= lr * error.mean()
		b -= lr * (error * x_norm).mean()

	a_original = a * (y.max() - y.min()) + y.min() - b * (x.min() / (x.max() - x.min())) * (y.max() - y.min())
	b_original = b * (y.max() - y.min()) / (x.max() - x.min())

	error_all_sum = 0
	for i in error:
		error_all_sum += i ** 2

	mse_error = error_all_sum / len(df["price"])

	print("El error cuadratico medio es: ", mse_error)

	print(f"Theta propio (gradiente descendente, desnormalizado): {a_original} y {b_original}")

	data = {"theta0": a_original, "theta1": b_original}
	with open("./OuterCore/ft_linear_regressin/trained_data.json", "w") as archivo:
		json.dump(data, archivo)


if __name__ == "__main__":
	main()