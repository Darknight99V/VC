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

Una posible solución en la que se pensó para solventar este problema fue guardar las posiciones de las monedas detectadas entre frame y frame. Y más adelante, esta idea fue complementada con otras soluciones extras proporcionadas por la IA. Entre ellas, se encuentra realizar **tracking** de las monedas detectadas. Sin embargo, no se implementó por falta de tiempo. 

<br><br>

## Tarea 2

Para el abordamiento de esta tarea se siguió el siguiente procedimiento.

1. Segmentación de las imágenes de entrenamiento
2. Extracción de características
3. Selección y entrenamiento de un clasificador de la librería sklearn
4. Testeo con la imagen de test proporcionada
5. Análisis de los resultados y obtención de métricas

<br><br>

#### Segmentación de las imágenes de entrenamiento

En un inicio se intentó utilizar ecualizado del histograma para reducir el ruido de las imágenes, pero no resultaba efectivo para las tres. Entonces, se procedió primero con el umbralizado, para el
cual se utilizó *Otsu* y además se aplicó un suavizado Gaussiano previamente en un intento de reducir el ruido. Este proceso se realiza en la función *blur_and_otsu()* y utilizó con todas las imágenes de entrenamiento. Finalmente con la que mejor funcionó fue con el alquitrán, como se puede observar en la siguiente imagen:

![Imagen procesada de alquitrán](https://github.com/Darknight99V/VC/blob/master/P3/images/tar-processed.png)

Para las dos imágenes restantes se combinó el umbralizado con otras técnicas. En el caso de los fragmentos se probó de nuevo a realizar el equalizado del histograma, pero esta vez se suavizó la imagen con el filtro de la mediana previamente, lo que se realiza en la función *histogram()*. Después se umbralizó la imagen resultante con un valor de umbral de 15. Los resultados fueron los siguientes:

![Imagen procesada de fragmentos](https://github.com/Darknight99V/VC/blob/master/P3/images/fra-processed.png)

Finalmente, la imagen de los pellets presentó dificultades debido a la iluminación de la imagen. Por ello, se hizo una consulta a la IA, que sugirió realizar una corrección del brillo de la imagen con la función *divide()* de OpenCV. Tras ello se siguió con el umbralizado, se utilizó en este caso un valor de umbral de 245. Este procesado resultó de la siguiente manera:

![Imagen procesada de pellet](https://github.com/Darknight99V/VC/blob/master/P3/images/pel-processed.png)

Y con esto concluye el proceso de segmentación de las imágenes de entrenamiento.

<br><br>

#### Extracción de características

El proceso de extracción de características se recoge en la función *extract_characteristics()*. Dicha función procede en primer lugar a extraer los contornos externos de la imagen procesada que se pasa como parámetro. Seguidamente, se recorren los contornos detectados y se calculan una serie de características geométricas, que son las siguientes:

- área
- perímetro
- compacidad
- ajuste de la elipse

Además, se utilizan el valor del área y la longitud del contorno (len(c)) para hacer un pequeño filtrado adicional en caso de que se haya detectado algo de ruido como un contorno más. De manera adicional, se ha añadido como caracrerística el color de la partícula, que se obtuvo calculando el centro del objeto. Todas estas características se recogen en un vector, que a su vez se almacena en el array **features_matrix**.

La extracción se realiza en cada una de las imagenes de entrenamiento y luego se combina todo en una única matriz. En este momento también se le ascocia a cada clase una *label*:
- FRA -> 0
- PEL -> 1
- TAR -> 2

<br><br>

#### Selección y entrenamiento de un clasificador de la librería sklearn

Se decidió utilizar el clasificador de Random Forest, que se basa en la generación de árboles de decisión a partir de la segmentación de los datos de entrenamiento. El entrenamiento fue el siguiente:

```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# entrenamos el clasificador con los datos obtenidos
clf.fit(final_matrix, labels)
```
<br><br>

#### Testeo con la imagen de test proporcionada

Para el desarrollo de este apartado, se planteó un primer acercamiento en el que se probaba a detectar los objetos de la imagen de testo (ver imagen del apartado) con un procesamiento muy básico, umbralizando con un método de **gaussiano adaptativo**. Seguidamente, al igual que con las imágenes de entrenamiento, se extrajeron las características geométricas previamente definidas, además del color del objeto. Luego, extrayendo datos del ficherp [Mps_test_bbs.csv](https://github.com/Darknight99V/VC/blob/master/P3/MPs_test_bbs.csv), se correlacionaron las coordenadas del centro calculado del objeto con las coordenadas del **bounding rectangle** del fichero para determinar que el objeto detectado pertenece a una determinada fila del archivo .csv y poder así determinar con certeza su clase. Por último, se fue pasando el vector de características calculado al clasificador y comparando su respuesta con la fila correspondiente en el fichero .csv. 

![Imagen de test](https://github.com/Darknight99V/VC/blob/master/P3/MPs_test.jpg)

Sin embargo, tras unos resultados muy pobres, decidimos indagar un poco en qué estaba pasando y nos fijamos que muchos objetos de la imagen de test no se detectaban correctamente. Es por ello que procedimos a usar la técnica de particionado de la imagen, decidiendo partirla en dos columnas y 3 filas, obteniendo así 6 imágenes nuevas. Con el fin de completar ágilmente y sin errores este nuevo planteamiento, nos apoyamos en la IA para segregar el código en funciones y partir la imagen en 6 fragmentos. El resultado obtenido de dicha interacción fue el siguiente:

```python
def preprocesar(grey_img, block_size=11, C=2, thr=120):
    processed = 255 - cv2.adaptiveThreshold(
        grey_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    _, imth = cv2.threshold(processed, thr, 255, cv2.THRESH_BINARY)
    return imth

def trocear_imagen(img, n_cols=2, n_rows=3):
    h, w = img.shape[:2]
    w_step = w // n_cols
    h_step = h // n_rows

    trozos = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * w_step
            y0 = r * h_step
            # el último trozo de cada fila/columna absorbe el resto
            x1 = (c + 1) * w_step if c < n_cols - 1 else w
            y1 = (r + 1) * h_step if r < n_rows - 1 else h
            sub = img[y0:y1, x0:x1].copy()
            nombre = f"fila{r}_col{c}"
            trozos.append((nombre, sub, x0, y0))
    return trozos


def evaluar_subimagen(clf, sub_bgr, x0, y0, csv_labels, csv_results,
                      area_filter=20, clon_filter=40,
                      block_size=11, C=2, thr=120, verbose=False):
    # preprocesar el trozo
    grey = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2GRAY)
    imth = preprocesar(grey, block_size, C, thr)

    matrix, coords_locales = extract_characteristics(
        imth, sub_bgr.copy(), area_filter, clon_filter
    )

    if len(matrix) == 0:
        return {"nombre": None, "n": 0, "acc": None,
                "y_real": [], "y_pred": [], "matrix": matrix}

    y_real, y_pred = [], []

    for i in range(len(matrix)):
        cy_l, cx_l = coords_locales[i]
        cy_g = cy_l + y0
        cx_g = cx_l + x0

        real_class = 0
        for j, (x_min, x_max, y_min, y_max) in enumerate(csv_results):
            if x_min <= cx_g <= x_max and y_min <= cy_g <= y_max:
                real_class = csv_labels[j]
                break

        # predicción
        pred = int(clf.predict(matrix[i].reshape(1, -1))[0])

        y_real.append(real_class)
        y_pred.append(pred)

        if verbose == True:
            print(f"    objeto {i:3d}  global=({cy_g},{cx_g})  "
                  f"real={real_class}  pred={pred}  "
                  f"{'ACIERTO' if pred == real_class else 'FALLO'}")

    acc = accuracy_score(y_real, y_pred) if y_real else None
    return {"nombre": None, "n": len(matrix), "acc": acc,
            "y_real": y_real, "y_pred": y_pred, "matrix": matrix}

def evaluar_imagen_completa(clf, ruta_img, csv_labels, csv_results,
                            n_cols=2, n_rows=3,
                            area_filter=20, clon_filter=40,
                            block_size=11, C=2, thr=120,
                            verbose=False):
    img = cv2.imread(ruta_img)
    if img is None:
        raise FileNotFoundError(ruta_img)

    trozos = trocear_imagen(img, n_cols, n_rows)

    resultados_por_trozo = []
    y_real_all, y_pred_all = [], []

    for nombre, sub, x0, y0 in trozos:
        if verbose:
            print(f"\n=== Subimagen {nombre}  offset=({x0},{y0})  "
                  f"shape={sub.shape} ===")

        res = evaluar_subimagen(
            clf, sub, x0, y0, csv_labels, csv_results,
            area_filter=area_filter, clon_filter=clon_filter,
            block_size=block_size, C=C, thr=thr, verbose=False
        )
        res["nombre"] = nombre
        res["offset"] = (x0, y0)
        resultados_por_trozo.append(res)

        y_real_all.extend(res["y_real"])
        y_pred_all.extend(res["y_pred"])

        if verbose == True:
            if res["acc"] is None:
                print(f"  -> sin objetos en este trozo")
            else:
                print(f"  -> n={res['n']}  acc={res['acc']:.3f}")

    # resumen global
    print("\n================ RESUMEN GLOBAL ================")
    print(f"Total objetos evaluados: {len(y_real_all)}")
    if y_real_all:
        print(f"Accuracy global: {accuracy_score(y_real_all, y_pred_all):.3f}")
        print(f"Distribución y_real: {Counter(y_real_all)}")
        print(f"Distribución y_pred: {Counter(y_pred_all)}")
        print("Matriz de confusión [filas=real, cols=pred]:")
        print(confusion_matrix(y_real_all, y_pred_all))
    else:
        print("No se ha detectado ningún objeto. Revisa filtros/preprocesado.")

    return resultados_por_trozo, y_real_all, y_pred_all


test_labels, test_results = csv_processing()

# 3) evaluar la imagen completa troceada
res_trozos, y_real_all, y_pred_all = evaluar_imagen_completa(
    clf, "MPs_test.jpg", test_labels, test_results,
    n_cols=2, n_rows=3,
    area_filter=30, clon_filter=20,   
    block_size=11, C=2, thr=120,
    verbose=False
)

metrics(y_pred_all, y_real_all)
```

<br><br>

#### Análisis de los resultados y obtención de métricas

Para esta parte del proceso de desarrolló una función que realiza estos cálculos y además devuelve la matriz de confusión, que en este caso fue:

![Matriz de confusión](https://github.com/Darknight99V/VC/blob/master/P3/images/conf-matrix.png)

Como se puede observar, el clasificador determina con una precisión aceptable los fragmentos y los confunde, en todo caso, con el alquitrán, posiblemente debido a sus similitudes en cuanto a forma. A continuación, los pellets los detectaba sin éxito alguno y desconocemos la causa de este fenómeno. Con los fragmentos de alquitrán se puede observar un comportamiento similar a los fragmentos pero invertido, confundiendo el alquitrán por fragmentos. Esto, probablemente, se dé por sus características físicas entre ambos tipos. 

Por último, cabe destacar que en dicha clasificación no figuran todos los objetos de la imagen, debido a que se necesita un mejor procesado de las imágenes particionadas para detectar con mayor precisión los objetos. 


