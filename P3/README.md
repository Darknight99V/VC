# Práctica 3 de Visión por Computador

## Componentes del grupo:

- Iván Luján Moreno ([@Darknight99V](https://github.com/Darknight99V))
- Carla Gómez García ([@carlagomez22](https://github.com/carlagomez22))

<br><br>

## Introducción
Esta práctica consta de dos tareas, la primera de conteo de monedas localizando sus contornos y obteniendo características geométricas. La segunda consiste en la clasificación de microplásticos
en una imagen utilizando nuevamente características geométricas. Esta última tarea se basa en el trabajo [SMACC: A System for Microplastics Automatic Counting and Classification](https://ieeexplore.ieee.org/document/8976153)
donde en vez de tres clases de microplásticos (Fragmentos, Pellets y Alquitrán) se cuenta con cinco clases y se trabaja con varios clasificadores, entre los que se encuentran SVM y Random Forest.
Por otro lado cabe destacar que en ambas tareas está *prohibido* el uso de modelos de aprendizaje profundo.

<br><br>

## Tarea 1
Como se acaba de mencionar, esta tarea consiste en contar monedas, primero en una imagen ideal y luego en un vídeo en vivo. Se propone en el enunciado encontar la equivalencia entre mm y píxeles
haciendo, por ejemplo, clic en una de las monedas. Y una vez obtenida la relación calcular la cantidad de dinero que hay en la foto.

En primer lugar, se buscó el diámetro en mm de todas las monedas que se obtuvo de la página de la [Comunidad Autónoma de la Región de Murcia](https://www.carm.es/web/pagina?IDCONTENIDO=718&IDTIPO=100&RASTRO=c285$m) y se dispuso la información
en los siguientes arrays:

```python
monedas_diámetro_mm = [25.75,23.25,24.25,22.25,19.75,21.25,18.75,16.26]
monedas_valor = [2, 1, 0.5, 0.20, 0.10, 0.05, 0.02, 0.01]
```
<br><br>

Los siguientes pasos tomados fueron umbralizar la imagen utilizando un valor umbral de 200 y con el método *findContours()* de OpenCV se obtuvieron los contornos exteriores de las monedas. Para obtener
el valor de la escala a partir de clicar la imagen se construyó una función, llamda *seleccionar_moneda()*, en la que principalmente se obtiene en qué contorno se está haciendo clic con el método *pointPolygonTest()*.

```python
escala = 0

def seleccionar_moneda(evento, x, y):
    global escala

    if evento == cv2.EVENT_LBUTTONDOWN:
        for i, contorno in enumerate(contornos_ideal):
            resultado = cv2.pointPolygonTest(contorno, (float(x), float(y)), False)

            if resultado >= 0:
                (_, _), radio = cv2.minEnclosingCircle(contornos_ideal[i])
                diametro_px = 2*radio

                escala = monedas_diámetro_mm[1] / diametro_px
                print(escala)

cv2.namedWindow("Imagen")
cv2.setMouseCallback("Imagen", seleccionar_moneda, img)
```

Cabe destacar que para el desarrollo de esta función se utilizó la *Inteligencia Artificial* para el manejo de eventos con OpenCV y el uso de la función pointPolygonTest().

Finalmente, una vez obtenida la escala, se procedió al cálculo del total de dinero presente en la imagen. Esto se hizo multiplicando los diámetros de los contornos obtenidos por la escala y luego
se comparaban con los diámetros reales del array **monedas_diámetro_mm**. Si hay coincidencia, el índice correspondiente se utiliza en el array **monedas_valor** para obtener el valor de la moneda.
En la variable acumuladora **total_value** se va guardando la suma del dinero, que finalmente era de 3.88€.

<br><br>

A continuación, se procedió con el demostrador en vivo del conteo de monedas. En una primera instancia se procedió de la misma manera que con la imagen ideal para la detección de contornos y el umbralizado,
pero los resultados obtenidos no detectaban del todo bien las monedas. Entonces se decidió probar con la Transformada de Hough y antes se aplica al fotograma un suavizado Gaussiano para eliminar un poco de ruido.
Después, para calcular la escala se hizo en función al radio y se cogió como referencia la moneda de 2€ que es la de mayor tamaño entre los círculos detectados por Hough. A la hora de buscar el valor de la moneda se utilizó un valor
de tolerancia que tiene en cuenta el error que pueda surgir en los cálculos. La suma de dinero se va acumulando en la variable **total_value**.

El problema principal que se encontró es las variaciones en la detección de monedas que existe de un frame a otro, por lo que el conteo varía de frame a frame. Además, seguramente por el valor de la tolerancia, se confunden algunas monedas
que son próximas en tamaño como las de 5 y 2 céntimos.

(COMPLETAR)

Una posible solución que se nos ocurrió para solventar este problema fue guardar las monedas detectadas entre frame y frame. Y más adelante, esta idea fue complementada con otras soluciones extras proporcionadas por la IA. Entre ellas, se encuentra un **tracking** de las monedas detectadas. Sin embargo, no se implementó por falta de tiempo. 

<br><br>

## Tarea 2

(especificar primero el procedimiento seguido)
