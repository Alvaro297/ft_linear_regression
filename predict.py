import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

def main():
	df = pd.read_csv("./data.csv")
	x = df['km']
	y = df['price']

	with open("./OuterCore/ft_linear_regressin/trained_data.json", "r") as archivo:
		json_trained = json.load(archivo)
	
	a = json_trained['theta0']
	b = json_trained['theta1']
	try:
		km_prediction = int(input("Introduce los kms del coche: "))
		if not km_prediction:
			km_prediction = 0
	except ValueError:
		km_prediction = 0
	y_pred_user = a + b * km_prediction
	y_pred = a + b * x
	
	print("El precio del coche es: ", y_pred_user)

	x_list = list(x) + [km_prediction]
	y_pred_list = list(y_pred) + [y_pred_user]
	
	# Ordenar por x (usando zip para mantener correspondencia)
	combined = sorted(zip(x_list, y_pred_list))
	x_combined, y_pred_combined = zip(*combined)
	
	plt.scatter(x, y, color='blue', label='Datos reales')
	plt.scatter(km_prediction, y_pred_user, color="yellow", label='Precio predicho')
	plt.plot(x_combined, y_pred_combined, color='red', label='Línea de regresión')
	plt.xlabel('km')
	plt.ylabel('price')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()