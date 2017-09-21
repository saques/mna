En el directorio informe/ se encuentra tanto la versión PDF del informe como el código fuente del mismo.

En el directorio python/ se encontrarán las implementaciones realizadas:

	- En lib.py, se podrá encontrar el algoritmo QR implementado, como así también las implementaciones 
	de soporte para el mismo (Matriz de Hessemberg, descomposición QR, reflectores de Householder).
	- kpca.py y pca.py contienen clases que sirven de interfaz a las implementaciones de los respectivos métodos.
	- cam.py provee una interfaz en tiempo real del reconocimiento de caras.
	
En el directorio orl_faces/ se encontrarán, en las primeras 5 carpetas, las imágenes que conforman la base 
de datos construiída por el equipo. Las comparaciones del informe han contrastado dicho set de imágenes con los
primeros 5 indivíduos de la base de datos de Cambridge.