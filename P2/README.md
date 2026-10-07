# Práctica 2 de Visión por Computador

## Componentes del grupo:

- Iván Luján Moreno ([@Darknight99V](https://github.com/Darknight99V))
- Carla Gómez García ([@carlagomez22](https://github.com/carlagomez22))

<br><br>

## Tarea 1

**Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil. Resalta con alguna primitiva gráfica en la imagen de Canny las filas que cumplen dicha condición.** 

Para la realización de la tarea, se ha seguido un proceso similar al que se hizo en el código de ejemplo con el conteo por columnas y se ha partido de la imagen del mandril procesada con el detector de bordes *Canny*. Para ello, se ha utilizado la función *reduce* de OpenCV, que hace que la imagen de Canny se compacte en una columna sumando todos los valores de cada fila. Luego se normaliza en base al número de columnas para obtener el número de píxeles blancos por fila. Por último, se obtiene el máximo y se filran los valores que sean iguales o superiores a *0.9xmaxfil*.

Los resultados obtenidos se ven reflejados en la siguiente imagen:

![Gráfica tarea 1](/P2/resultados/ouput-tarea1.png)

En la imagen se puede observar la salida de Canny a la izquierda y, a la derecha, un histograma que muestra la proporción de píxeles blancos contados en cada fila. Además, los puntos dibujados en rojo son aquellas filas que superaron el *0.9xmaxfil*.


<br><br>

## Tarea 2

**Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. Visualiza los resultados obtenidos para la imagen (o una de tu elección) con Canny y Sobe tras umbralizar ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?**

En primer lugar, se umbralizó la imagen de Sobel utlizando un valor umbral arbitrario de 100. Después, se procedió al conteo de píxeles blancos por filas y por columnas, tal y como se había hecho con la imagen de Canny. Tras ello, se obtuvieron los máximos y se filtraron las filas y columnas que igualaran o superaran el 0.9 de esos máxmimos. Para mostrar los resultados sobre la imagen del mandril se han utilizado líneas azules para las columnas y líneas rojas para las columnas. Además, se ha mostrado en otra imagen las filas y columnas que tienen en común Canny y Sobel tras umbralizar.

Estas son las imágenes con los resultados:

![Gráfica 1 tarea 2](https://github.com/Darknight99V/VC/blob/master/P2/resultados/output1-tarea2.png)
![Gráfica 2 tarea 2](https://github.com/Darknight99V/VC/blob/master/P2/resultados/output2-tarea2.png)
![Gráfica 3 tarea 2](https://github.com/Darknight99V/VC/blob/master/P2/resultados/output3-tarea2.png)


Canny detecta mejor bordes verticales y Sobel los bordes horizontales. También se ve que solo tienen pocas filas en común en la última imagen.




