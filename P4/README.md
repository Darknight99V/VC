# Práctica 4 y Práctica 4b de Visión por Computador

## Componentes del grupo:

- Iván Luján Moreno ([@Darknight99V](https://github.com/Darknight99V))
- Carla Gómez García ([@carlagomez22](https://github.com/carlagomez22))

<br><br>

## Introducción

La práctica consta de dos tareas principales. La primera requiere de los siguientes puntos:

- Detección y seguimientos de personas y vehículos presentes.
- Detección de matrículas de los vehículos presentes.
- Cuenta total de cada clase.
- Volcado en disco de un vídeo que muestre los resultados.
- Generación de un archivo CSV con el resultado de la detección y seguimiento, con al menos los siguientes campos: fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matrícula_en_su_caso, confianza, mx1,my1,mx2,my2, texto_matricula

  Adicionalmente, realizamos los siguientes apartados opcionales:
  - Determinación del flujo de personas y vehículos en el vídeo de test en las distintas direcciones.
  - Evaluación de alternativas para la detección de matrículas: una basada en YOLO y otra en RT-DETR.


Para la Práctica 4b, se piden los siguientes apartados:
  - Completar el CSV con la adquisición de texto de cada matrícula detectada.
  - Realizar una comparativa entre dos OCR, uno clásico y un VLM, en términos de tiempo de inferencia y tasa de acierto.

<br><br>

 ## Tarea 1

### Detección y seguimiento de personas y vehículos

Para la resolución se esta tarea, simplemente se usó el modelo preentrenado de YOLO (concretamente: yolo11n.pt). En su uso se especificaron las clases que se iban a detectar y además se usó la función de *track()* al ejecutar el modelo para poder hacer el seguimiento. Las clases utilizadas fueron las siguientes: *"person", "bicycle", "car", "motorbike", "bus" y "truck"*. A continuación se muestra cómo se usó el modelo:

```python
model.track(frame, persist=True, classes=[0,1,2,3,5,7], verbose=False)
```

Los resultados de estas detecciones se verán en el vídeo final que contiene todas las anotaciones de la práctica.

<br><br>

### Detección de matrículas

Para poder detectar matrículas, se propone en la práctica el entrenamiento de un modelo de YOLO. Para la obtención del dataset se dan varias opciones, pero finalmente el alumnado de la asignatura se ha coordinado para generar un pequeño [dataset](https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI?usp=sharing) de alrededor de 2000 imágenes de matrículas. Dichas imágenes se encuentran en formato JPG en la carpeta *images* y sus respectivas anotaciones en la carpeta *annotations*.

Las anotaciones se realizaron utilizando un script de python ([script](https://drive.google.com/file/d/1Y8OJj8H2FHFxNMdYG62AEV5gN6-EbCAU/view?usp=drive_link)) desarrollado por uno de los compañeros. Estas anotaciones siguen la notación de YOLO normalizada y se guardaron en formato JSON:

```json
[
  {
    "rectangle": [
      0.6841878172588833,
      0.501269035532995,
      0.05208121827411161,
      0.02314720812182741
    ],
    "plate": "2221FXK"
  },
  {
    "rectangle": [
      0.4100761421319797,
      0.47446700507614215,
      0.02893401015228425,
      0.017461928934010162
    ],
    "plate": "None"
  }
]
```
 
 Como se puede observarm, las anotaciones poseen el texto de la matrícula y el *bounding box* de cada una de las matrículas mayoritariamente legibles o reconocibles en cada imagen. Si una matrícula era reconocible pero su contenido no se legible, ya fuera por estar parcialmente tapada por otro objeto o estar muy lejos, el valor de la matrícula se ponía a *None*.

<br><br>

Antes de comenzar el entrenamiento, las imágenes se separaron en 3 conjuntos: train, test y val. El porcentaje de imágenes que posee cada carpeta con respecto al total de imágenes que tiene el dataset, respectivamente, es el siguiente: 64%, 20% y 16%.

Además, también fue necesario para cumplir con el formato que requiere YOLO para el entrenamiento, convertir las anotaciones de formato JSON a TXT, lo que se hizo con un pequeño script. (Poner script) 

(especificar mejor los parámetros de entrenamiento del YOLO: épocas, qué modelo se entrenó, LR...)
(explicar también alguna gráfica relevante que tenga que ver con el desarrollo del entrenamiento, como la función de pérdida o la matriz de confusión)

<br><br>

### Conteo por clases y flujo de personas y vehículos

 Por otro lado, para el conteo total de cada clase y su flujo, se fue contando *frame* a *frame* los distintos objetos que detectaba el modelo de YOLO y se analizaba la clase y el ID de cada uno. 
 Se contaba con con el número de clase como clave y el nombre de la clase como valor, y un *set* que contenía los distintos IDs detectados para dicha clase, y evitaba IDs duplicados, para realizar el conteo.

 Estas son las estructuras en las que se apoyó el conteo
 ```python
classCounter = {"person": 0, "bicycle": 0, "car": 0, "motorbike": 0, "bus": 0, "truck": 0}
classNames = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
uniqueObjects = {nombre: set() for nombre in classNames.values()}
```

En este fragmento del código se recorren los resultados obtenidos después de ejecutar el modelo en el frame actual para realizar el conteo.
```python
        for result in results:
            for box in result.boxes:
                clase = int(box.cls.item())

                if clase not in classNames:
                    continue

                nombre = classNames[clase]
                classCounter[nombre] += 1

                if box.id is not None:
                    id_objeto = int(box.id.item())
                    uniqueObjects[nombre].add(id_objeto)
```


 Asimismo, para determinar el flujo, debido a que los objetos en el vídeo se encontraban en movimiento, se hizo una aproximación teniendo en cuenta su primera y última posición detectada. Dependiendo de la trayectoria del objeto, se determinaba por cual de los 4 cuadrantes de la imagen había "salido" el objeto. (Explicar mejor relacionando con el código y las estructuras relacionadas)

```python
class Tracked_Object:

    def __init__(self, id : int, type : int) -> None:
        self.id = id
        self.points = [] # los distintos puntos asociados a este objeto
        self.trayectory_x = 0
        self.trayectory_y = 0
        self.type = type

    def add_point(self, xy) -> None:
        x, y = xy[0], xy[1]
        self.points.append((x, y))

    def calculate_trayectory(self) -> None:
        self.trayectory_x = self.points[-1][0] - self.points[0][0]
        self.trayectory_y = self.points[-1][1] - self.points[0][1]

    def __str__(self) -> str:
        return f"ID:{self.id}, type:{self.type}"

class Image_Section: 

    def __init__(self, section_name : str) -> None:
        self.name = section_name
        self.class_names = {0:"person", 1:"bicycle", 2:"car", 3:"motorbike", 5:"bus", 7:"truck"}
        self.detections = {} # par clave-valor clase : cuenta

    def add_obj(self, obj : Tracked_Object) -> None:
        self.detections[obj.type] = self.detections.get(obj.type, 0) + 1

    def __str__(self):
        total = sum(self.detections.values())
        string = f"Esta sección de {self.name} contiene {total} objetos:\n"
        for key, count in self.detections.items():
            string += f"- {self.class_names.get(key, f'clase_{key}')} : {count}\n"
        return string
```

(de este cacho de código si quieres quédate con lo que solo tiene que ver con lo del tracking y ya después al final se pone entero y ya verá el contexto y tal)
```python
#video
filename = "coches.MP4"
ANCHO_MAX = 1280

#clases que se van a detectar
classCounter = {"person": 0, "bicycle": 0, "car": 0, "motorbike": 0, "bus": 0, "truck": 0}
classNames = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
uniqueObjects = {nombre: set() for nombre in classNames.values()}

#carga de los modelos
model = YOLO('yolo11n.pt')
last_known_position = {} 
objects = {} # clave-par id:obj
coches = cv2.VideoCapture(filename)

while(True):
    ret, frame = coches.read()

    if ret:
        alto, ancho = frame.shape[:2]
        if ancho > ANCHO_MAX:
            factor = ANCHO_MAX / ancho
            frame = cv2.resize(frame, (ANCHO_MAX, int(alto * factor)))
        alto, ancho = frame.shape[:2]

        # Reiniciar contadores en cada frame
        classCounter = {nombre: 0 for nombre in classNames.values()}   

        results = model.track(frame, persist=True, classes=[0,1,2,3,5,7], verbose=False)
        frame = results[0].plot()

        #recorrer resultados
        for result in results:
            for box in result.boxes:
                clase = int(box.cls.item())

                if clase not in classNames:
                    continue

                nombre = classNames[clase]
                classCounter[nombre] += 1

                if box.id is not None:
                    id_objeto = int(box.id.item())
                    uniqueObjects[nombre].add(id_objeto)
            
            ids = result.boxes.id.int().cpu().tolist()
            boxes_xywh = result.boxes.xywh.cpu().tolist()
            types = result.boxes.cls.int().cpu().tolist()

            for id, coords, type in zip(ids, boxes_xywh, types):
                if id not in objects:
                    objects[id] = Tracked_Object(id, type)
                objects[id].add_point(coords)

            #Dibujar contador
        objetos = []

        for nombre in classNames.values():
            actual = classCounter[nombre]
            total = len(uniqueObjects[nombre])

            objetos.append(f"{nombre}: {actual} (Total: {total})")

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.55
        grosor = 2
        margen = 10
        separacion = 25

        # Calcular dimensiones del panel
        dimensiones = [
            cv2.getTextSize(texto, fuente, escala, grosor)[0]
            for texto in objetos
        ]

        ancho_panel = max(w for w, h in dimensiones) + 20
        alto_panel = len(objetos) * separacion + 15

        # Situar panel en esquina superior derecha
        x_panel = max(0, ancho - ancho_panel - margen)
        y_panel = margen

        # Dibujar fondo negro
        cv2.rectangle(
            frame,
            (x_panel, y_panel),
            (x_panel + ancho_panel, y_panel + alto_panel),
            (0, 0, 0),
            -1
        )

        # Dibujar cada linea
        for i, texto in enumerate(objetos):

            x = x_panel + 10
            y = y_panel + 20 + i * separacion

            cv2.putText(
                frame,
                texto,
                (x, y),
                fuente,
                escala,
                (0, 255, 0),
                grosor,
                cv2.LINE_AA
            )
        cv2.imshow('Vid', frame)

    # Detenemos pulsado ESC
    if cv2.waitKey(20) == 27:
        break

coches.release()

lu = Image_Section("arriba izquierda")
ld = Image_Section("abajo izquierda")
ru = Image_Section("arriba derecha")
rd = Image_Section("abajo derecha")

for id in objects.keys():
    obj = objects[id]
    obj.calculate_trayectory()

    if obj.trayectory_x < 0 and obj.trayectory_y < 0:
        ld.add_obj(obj)
    elif obj.trayectory_x < 0 and obj.trayectory_y > 0:
        lu.add_obj(obj)
    elif obj.trayectory_x > 0 and obj.trayectory_y > 0:
        ru.add_obj(obj)
    else:
        rd.add_obj(obj)

print(objects)

print(lu)
print(ld)
print(ru)
print(rd)
```

Finalmente, se utilizaron ambos modelos (el preentrenado con las clases seleccionadas y el que se entrenó con matrículas) sobre el vídeo de prueba proporcionado por el profesorado. El código también integra el conteo por clases y el flujo de personas y vehículos anteriormente expuestos.

(poner código)

### Generación del CSV y volcado del vídeo en memoria
(poner código de generación  de CSV asi como enlace a este CSV y una captura a modo de preview y decir que el volcado de memoria se hizo utilizando openCV, poner también gif del vídeo resultante)

### Comparación entre YOLO y RT-DETR
(modelos entrenados en ambos casos, qué parámetros del entrenamiento se dejaron iguales, poner que se hizo en google colab y especificarlo en el YOLO también, mostrar gráficas relevantes y explicarlas, etc)


## Tarea 2 (Práctica 4b)

 A la hora de completar esta tarea, para comparar ambos modelos de OCR, usamos un pequeño conjunto de X imágenes para una comparativa con datos significativos. 

 Por parte del OCR clásico, se obtuvieron los siguientes resultados:


 Por parte del OCR VLM, se obtuvieron los siguientes resultados:

 Como se puede observar, el OCR clásico destaca por su .... Por otro lado, el OCR VLM destaca por .... 

  
