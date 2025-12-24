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

	X_ones = np.c_[np.ones(x_norm.shape[0]), x_norm]

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

	theta = np.linalg.inv(X_ones.T.dot(X_ones)).dot(X_ones.T).dot(y_norm)  # Ahora usa y_norm
	print("Theta (normalizado):", theta)
	print(f"Theta propio (gradiente descendente, desnormalizado): {a_original} y {b_original}")
	#Crear un json con theta
	data = {"theta0": a_original, "theta1": b_original}
	with open("./OuterCore/ft_linear_regressin/trained_data.json", "w") as archivo:
		json.dump(data, archivo)
	# Predicciones (En el predict.py)
	y_pred = a_original + b_original * x
	
	# Gráfico
	plt.scatter(x, y, color='blue', label='Datos reales')
	plt.plot(x, y_pred, color='red', label='Línea de regresión')
	plt.xlabel('km')
	plt.ylabel('price')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()