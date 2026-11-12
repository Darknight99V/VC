# Práctica 4 de Visión por Computador

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

  Adicionalmente, realizamos los apartados opcionales:
  - Determinación del flujo de personas y vehículos en el vídeo de test en las distintas direcciones.
  - Evaluación de alternativas para la detección de matrículas: una basada en YOLO y otra en RRT-DETR.

  Para la segunda tarea, se piden los siguientes apartados:
  - Completar el CSV con la adquisición de texto de cada matrícula detectada.
  - Realizar una comparativa entre dos OCR, uno clásico y un VLM, en términos de tiempo de inferencia y tasa de acierto.

<br><br>

 ## Tarea 1

 Para la resolución de esta tarea, cogimos el vídeo de ejemplo y usamos el modelo pre entrenado de YOLO para detectar y hacer seguimiento de las personas y vehículos hemos usado el *track()* que incluye YOLO. Las clases que se han usado para detectar y seguir han sido las siguientes, respectivamente: *"person", "bicycle", "car", "motorbike", "bus" y "truck"*.

```python
model.track(frame, persist=True, classes=[0,1,2,3,5,7], verbose=False)
```

 Los resultados de estas detecciones se verán en el vídeo final que contiene todas las anotaciones.
 
 Para la parte de detección de matrículas, el alumnado de VC de la ULPGC se coordinó para generar un pequeño [dataset](https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI?usp=sharing) de ~2k imágenes en formato JPG con anotaciones en formato JSON de matrículas de vehículos. 
 
 Las anotaciones poseen el texto de la matrícula y el *bounding box* de cada una de las matrículas mayoritariamente legibles o reconocibles en cada imagen. Si una matrícula era reconocible pero su contenido no se legible, ya fuera por estar parcialmente tapada por otro objeto o estar muy lejos, entre otros, el valor de la matrícula se ponía a None. Todas estas imágenes se encuentran en una sola carpeta llamada *images* y las anotaciones en la carpeta *annotations*.
 
 Es por ello que separamos las imágenes en 3 conjuntos: train, test y val. El porcentaje de imágenes que posee cada carpeta con respecto al total de imágenes que tiene el dataset, respectivamente, es el siguiente: 64%, 20% y 16%.

 También, con un pequeño script, pasamos las anotaciones en formato JSON a TXT con formato YOLO. Este formato es el siguiente: clase x_centro y_centro ancho alto. 

 Por otro lado, para el conteo total de cada clase y su flujo, fuimos contando *frame* a *frame* los distintos objetos que detectaba el modelo de YOLO y analizábamos la clase y el ID de cada uno. Con un diccionario con las clases como llave y un *set* que contenía los distintos IDs detectados para dicha clase, fuimos capaces de hacer un conteo. 
 
 Asimismo, para determinar el flujo, debido a que los objetos en el vídeo se encontraban en movimiento, se hizo una aproximación teniendo en cuenta su primera y última posición detectada. Dependiendo de la trayectoria del objeto, se determinaba por cual de los 4 cuadrantes de la imagen había "salido" el objeto.

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

def compute_iou(box1, box2):
    # Calcula el Intersection over Union entre dos cajas [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

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

        # Dibujar cada lÃnea
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

Por último, con el modelo de YOLO entrenado para detectar matrículas, usábamos éste junto con el modelo pre entrenado para hacer la detección tanto de los vehículos y personas como de las matrículas. Y a su vez, se volcaron estos contenidos en un CSV y en un vídeo para mostrar los resultados. 

## Tarea 2

 A la hora de completar esta tarea, para comparar ambos modelos de OCR, usamos un pequeño conjunto de X imágenes para una comparativa con datos significativos. 

 Por parte del OCR clásico, se obtuvieron los siguientes resultados:


 Por parte del OCR VLM, se obtuvieron los siguientes resultados:

 Como se puede observar, el OCR clásico destaca por su .... Por otro lado, el OCR VLM destaca por .... 

  
