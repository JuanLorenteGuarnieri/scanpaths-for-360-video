import os
import json

# # Directorio donde se encuentran los archivos .scanpaths
# directorio = 'output_scanpaths'

# # Diccionario para clasificar los archivos por tipo
# archivos_por_tipo = {}

# # Recorrer los archivos en el directorio
# for archivo in os.listdir(directorio):
#     if archivo.endswith('.scanpaths'):
#         # Extraer el número, _N10, y el tipo del nombre del archivo
#         partes = archivo.split('_N10_')
#         numero, tipo_con_extension = partes[0], partes[1]
#         tipo = tipo_con_extension.split('.scanpaths')[0]
        
#         # Agregar el archivo a la lista correspondiente a su tipo
#         if tipo not in archivos_por_tipo:
#             archivos_por_tipo[tipo] = []
#         archivos_por_tipo[tipo].append(os.path.join(directorio, archivo))

# # Para cada tipo, leer los archivos, combinar los scanpaths y guardarlos
# for tipo, archivos in archivos_por_tipo.items():
#     scanpaths_combinados = []

#     for archivo in archivos:
#         with open(archivo, 'r') as f:
#             # Asumiendo que los scanpaths están almacenados en formato JSON
#             scanpaths = json.load(f)
#             scanpaths_combinados.extend(scanpaths)
    
#     # Guardar los scanpaths combinados en un nuevo archivo
#     nombre_archivo_resultado = os.path.join(directorio, f"{tipo}.scanpaths")
#     with open(nombre_archivo_resultado, 'w') as f:
#         json.dump(scanpaths_combinados, f)

# print("Proceso completado.")

import csv
import os
import json

# Función para procesar un único archivo y obtener los scanpaths
def procesar_archivo(ruta_archivo):
    datos_agrupados = []
    grupo_actual = []
    frame_anterior = -1
    ultimo_frame_guardado = -8
    ajuste_realizado = False
    
    with open(ruta_archivo, 'r') as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            frame_actual = int(fila['frame'])
            
            if frame_actual != frame_anterior:
                if frame_anterior > 0 and frame_actual < frame_anterior:
                    datos_agrupados.append(grupo_actual)
                    grupo_actual = []
                    frame_anterior = -1
                    ultimo_frame_guardado = -8
                    ajuste_realizado = False
                
                diferencia = frame_actual - ultimo_frame_guardado
                
                if diferencia >= 8 or (diferencia > 0 and not ajuste_realizado):
                    grupo_actual.append([float(fila['v']), float(fila['u'])])
                    if not ajuste_realizado and diferencia > 0 and diferencia < 8:
                        ultimo_frame_guardado = frame_actual
                        ajuste_realizado = True
                    elif ajuste_realizado:
                        ultimo_frame_guardado += 8
                    else:
                        ultimo_frame_guardado = frame_actual
            
            frame_anterior = frame_actual
        
        if grupo_actual:
            datos_agrupados.append(grupo_actual)
    
    return datos_agrupados[:10]  # Retorna solo los primeros 10 grupos

# Directorio de los archivos
directorio = 'D:/TFG/datasets/D-SAV360/gaze_data'

# Números de los archivos específicos a procesar
numeros_especificos = ['0002', '0011', '1005', '1016', '2006', '2017', '4002', '4008', '5007', '5035']

# Recolecta 10 scanpaths de cada archivo
scanpaths_totales = []
# Procesa solo los archivos especificados
for numero in numeros_especificos:
    nombre_archivo = f'gaze_video_{numero}.csv'
    ruta_completa = os.path.join(directorio, nombre_archivo)
    if os.path.exists(ruta_completa):
        scanpaths = procesar_archivo(ruta_completa)
        scanpaths_totales.extend(scanpaths)
    else:
        print(f"El archivo {nombre_archivo} no existe.")

# Guarda los scanpaths totales en un archivo .scanpaths
ruta_guardado = 'output_scanpaths/ground_truth.scanpaths'
with open(ruta_guardado, 'w') as archivo_resultado:
    json.dump(scanpaths_totales, archivo_resultado)

print("Proceso completado.")

