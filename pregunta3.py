import cv2
import mediapipe as mp
import math
import time
#video descargado de  https://drive.google.com/file/d/11uErcxRdM-L-RxdqlOjzcn9W5QusyI63/view?usp=sharing
video = cv2.VideoCapture('videoplayback (online-video-cutter.com).mp4')

#Declarado e inicializado la solucion face_mesh de mediapipe
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils
contador_vocal_abierta = 0
contador_vocal_cerrada = 0
vocal_abierta = False
vocal_cerrada = False
estado = 'X'
inicio = 0
estado_actual = ''


while True:
    check, img = video.read() #Leyendo el video con opencv
    img = cv2.resize(img, (1000, 720)) #Redimiensionando el video con cv2

    if not check:
        break

    results = faceMesh.process(img) #Procesando el video con faceMesh de mediapipe
    h, w, _ = img.shape #EXtrayendo dimensiones del video

    if results:
        if not results.multi_face_landmarks:
            continue
        #Puntos extraidos de https://i.stack.imgur.com/wDgvV.png
        for face in results.multi_face_landmarks:
            #Declarando los puntos de la boca y normalizandolos con las dimensiones extraidas anteriormente

            # PUNTOS CENTRALES DE LOS LABIOS
            m1x, m1y = int((face.landmark[13].x) * w), int((face.landmark[13].y) * h)
            m2x, m2y = int((face.landmark[14].x) * w), int((face.landmark[14].y) * h)
            
            # PUNTOS DEL LADO DERECHO DE LOS LABIOS
            m3x, m3y = int((face.landmark[81].x) * w), int((face.landmark[81].y) * h)
            m4x, m4y = int((face.landmark[178].x) * w), int((face.landmark[178].y) * h)

             # PUNTOS DEL LADO IZQUIERDO DE LOS LABIOS
            m5x, m5y = int((face.landmark[311].x) * w), int((face.landmark[311].y) * h)
            m6x, m6y = int((face.landmark[402].x) * w), int((face.landmark[402].y) * h)

            # CALCULO DE LAS DISTANCIAS DE LOS PUNTOS CENTRALES, DERECHO E IZQUIERDO.
            distM = math.hypot(m1x - m2x, m1y - m2y)
            distM2 = math.hypot(m3x-m4x,m3y-m4y)
            distM3 = math.hypot(m5x-m6x,m5y-m6y)
            
            #EN esta parte se verifica apartir de las distancias si la boca se encuentra abierta o cerrada, esto con
            #el fin de contabilizar vocales abiertas cuando la boca este abierta y viceversa.
            #LA DISTANCIA MINIMA PARA CONSIDERAR BOCA ABIERTA ES DE 10
            if distM > 10 and distM2 > 10 and distM3 > 10:
                vocal_cerrada = True
                if vocal_abierta:
                    contador_vocal_abierta +=1 #Para que este contador sume antes la boca tendría que haber estado cerrada
                    vocal_abierta = False
                    vocal_cerrada = False
                cv2.rectangle(img, (100, 30), (390, 80), (0, 0, 255), -1)
                cv2.putText(img, 'BOCA ABIERTA', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            else:
                vocal_abierta = True
                if vocal_cerrada:
                    contador_vocal_cerrada +=1 #Para que este contador sume antes la boca tendría que haber estado abierta
                    vocal_abierta = False
                    vocal_cerrada = False
            
                cv2.rectangle(img, (100, 30), (390, 80), (255, 0, 0), -1)
                cv2.putText(img, 'BOCA CERRADA', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            '''
            EL flujo anterior para contar las vocales abiertsa y cerradas es el siguiente:
            Para una vocal abierta:
                Antes los labios tendrían que estar cerrados y luego abrirse, este flujo lo obtenemos con el flag vocal_abierta
                ya que al hacerse True en la linea 67 y luego en la segunda iteracion al entrar en la linea 60 entonces se considera como
                una vocal abierta
            Para una vocal cerrada:
                Antes los labios tendrían que estar abiertos y luego cerrarse, este flujo lo obtenemos con el flag vocal_cerrada
                ya que al hacerse True en la linea 59 y luego entrar al if en la linea 69 entonces se considera como que la boca hizo
                una vocal cerrada
            '''

            #Aqui mostramos los resultados de la vocal abierta y vocal cerrada y con OPENCV imprimimos rectangulos y texto en la pantalla
            texto = f'Vocales abiertas: {contador_vocal_abierta}'
            texto2 = f'Vocales cerradas: {contador_vocal_cerrada}'
            cv2.rectangle(img, (20,240), (340,120), (255,0,0), -1)
            cv2.putText(img, texto, (25,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
            cv2.putText(img, texto2, (25,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
    cv2.imshow('Detector', img)
    if cv2.waitKey(120) & 0xFF == ord('q'):  # Agrega un retraso de 120 milisegundos entre cada cuadro
        break

video.release()
cv2.destroyAllWindows()