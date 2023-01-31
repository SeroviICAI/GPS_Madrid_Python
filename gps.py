import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pandas as pd
from grafo import Grafo, INFTY
from typing import Tuple

PESO_PREDETERMINADO = 'distancia'
RADIO_APROXIMACION = 10000

COLORES_TIPOS = {"AUTOVIA": "#42BA64",
                 "AVENIDA": "#00AE8B",
                 "CARRETERA": "#009FAD",
                 "CALLEJON": "#008BC0",
                 "CAMINO": "#008BC0",
                 "ESTACION DE METRO": "#0075BE",
                 "PASADIZO": "#0075BE",
                 "PLAZUELA": "#0075BE",
                 "COLONIA": "#0075BE",
                 "OTRO": "#3D5BA8"}

# Velocidades en centímetros por minuto
VELOCIDADES_TIPOS = {"AUTOVIA": 166667,
                     "AVENIDA": 150000,
                     "CARRETERA": 116667,
                     "CALLEJON": 50000,
                     "CAMINO": 50000,
                     "ESTACION DE METRO": 33333,
                     "PASADIZO": 33333,
                     "PLAZUELA": 33333,
                     "COLONIA": 33333,
                     "OTRO": 83333}


def extraer_datos() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Extrayendo datos...")
    columnas_cruces = [
        # Información principal del nodo
        'Codigo de vía tratado',
        'Codigo de via que cruza o enlaza',
        # Posición
        'Coordenada X (Guia Urbana) cm (cruce)',
        'Coordenada Y (Guia Urbana) cm (cruce)',
    ]

    columnas_direcciones = [
        # Información principal del nodo
        'Codigo de via',
        'Direccion completa para el numero',
        'Literal de numeracion',
        # Información adicional del nodo
        'Clase de la via',
        'Nombre del distrito',
        'Nombre del barrio',
        'Codigo postal',
        # Posición del nodo
        'Coordenada X (Guia Urbana) cm',
        'Coordenada Y (Guia Urbana) cm'
    ]

    # Leer csv cruces
    df_cruces = pd.read_csv(filepath_or_buffer='cruces.csv',
                            delimiter=';',
                            usecols=columnas_cruces,
                            encoding='latin1',
                            dtype=str)

    # Cambiar nombres columnas
    df_cruces.rename(columns={'Codigo de vía tratado': 'Codigo de via',
                              'Codigo de via que cruza o enlaza': 'Codigo de via cruzada',
                              'Coordenada X (Guia Urbana) cm (cruce)': 'Coordenada X (Guia Urbana) cm',
                              'Coordenada Y (Guia Urbana) cm (cruce)': 'Coordenada Y (Guia Urbana) cm'},
                     inplace=True)
    df_cruces['Clase de la via'] = 'CRUCE'

    # Leer csv cruces
    df_direcciones = pd.read_csv(filepath_or_buffer='direcciones.csv',
                                 delimiter=';',
                                 usecols=columnas_direcciones,
                                 encoding='latin1',
                                 dtype=str)

    # Cambiamos posición de la columna 'Codigo de via cruzada' a 1
    columna_codigo_cruzada = df_cruces.pop('Codigo de via cruzada')
    df_cruces.insert(1, 'Codigo de via cruzada', columna_codigo_cruzada)

    # Cambiar nombres columnas y cambiar 'Nombre' a posición 2
    df_direcciones.rename(columns={'Direccion completa para el numero': 'Nombre',
                                   'Literal de numeracion': 'Numero'}, inplace=True)
    columna_nombres = df_direcciones.pop('Nombre')
    df_direcciones.insert(1, 'Nombre', columna_nombres)

    # Strip todas las strings
    df_cruces = df_cruces.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_direcciones = df_direcciones.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Convertir a entero todas las strings que puedan pasar a ser enteras
    df_cruces.iloc[:, 2::] = df_cruces.iloc[:, 2::].applymap(
        lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
    df_direcciones.iloc[:, 2::] = df_direcciones.iloc[:, 2::].applymap(
        lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)

    # Crear columna coordenadas
    df_cruces['Coordenadas cm'] = tuple(zip(df_cruces['Coordenada X (Guia Urbana) cm'],
                                            df_cruces['Coordenada Y (Guia Urbana) cm']))
    df_cruces.drop(columns=['Coordenada X (Guia Urbana) cm', 'Coordenada Y (Guia Urbana) cm'],
                   inplace=True)
    df_direcciones['Coordenadas cm'] = tuple(zip(df_direcciones['Coordenada X (Guia Urbana) cm'],
                                                 df_direcciones['Coordenada Y (Guia Urbana) cm']))
    df_direcciones.drop(columns=['Coordenada X (Guia Urbana) cm', 'Coordenada Y (Guia Urbana) cm'],
                        inplace=True)

    # Crear un código de cruce
    columna_codigos_cruces = df_cruces.index
    df_cruces.insert(1, 'Codigo de cruce', columna_codigos_cruces)

    # Eliminar los cruces duplicados
    df_cruces['TEMP'] = df_cruces[['Codigo de via', 'Codigo de via cruzada']].apply(sorted, axis=1)
    df_cruces = df_cruces.drop_duplicates(subset='TEMP')

    # Crear nombres para cada cruce
    df_cruces['Nombre'] = ('CRUCE, C-' + df_cruces['Codigo de cruce'].apply(str))
    columna_nombres = df_cruces.pop('Nombre')
    df_cruces.insert(2, 'Nombre', columna_nombres)
    df_cruces.drop_duplicates(subset='Nombre', keep="first", inplace=True)

    # Igualar el nombre de aquellos cruces que tienen coordenadas cercanas (glorietas)
    def aproximar(coordenadas):
        return tuple([coord // RADIO_APROXIMACION for coord in coordenadas])

    df_cruces['TEMP'] = df_cruces['Coordenadas cm'].apply(aproximar)
    coordenadas_unicas = df_cruces['TEMP'].unique()

    for coordenada in coordenadas_unicas:
        if len(glorieta := df_cruces[df_cruces['TEMP'] == coordenada].copy()) > 1:
            glorieta['Nombre'] = glorieta.iloc[0]['Nombre']
            glorieta['Coordenadas cm'] = [glorieta.iloc[0]['Coordenadas cm']] * len(glorieta)
            df_cruces.loc[df_cruces['TEMP'] == coordenada] = glorieta
    df_cruces = df_cruces.drop(columns='TEMP')

    # Asumimos que los cruces son no dirigidos:
    df_cruces_inv = df_cruces[['Codigo de via cruzada', 'Codigo de via']].copy()
    df_cruces_inv.columns = ['Codigo de via', 'Codigo de via cruzada']
    df_cruces_inv[df_cruces.columns.difference(['Codigo de via', 'Codigo de via cruzada'])] = df_cruces[
        df_cruces.columns.difference(['Codigo de via', 'Codigo de via cruzada'])]
    df_cruces = pd.concat([df_cruces, df_cruces_inv], axis=0)

    # Quitar ceros innecesarios de las direcciones
    df_direcciones['Nombre'] = df_direcciones['Nombre'].str.replace(r'\b0+(?=[^0])', '', regex=True)

    # Eliminar falsas coordenadas
    def es_una_tupla_entera(tupla):
        return all(isinstance(x, int) for x in tupla)

    df_direcciones['es_entera'] = df_direcciones['Coordenadas cm'].apply(es_una_tupla_entera)
    df_direcciones = df_direcciones[df_direcciones['es_entera'] == True]
    df_direcciones.drop(columns=['es_entera'], inplace=True)

    # Ordenar dataframes
    df_cruces.sort_values(by=['Codigo de via'], ascending=True, inplace=True, ignore_index=True)
    df_direcciones.sort_values(by=['Codigo de via', 'Nombre'], inplace=True, ignore_index=True)

    # Quitar direcciones con la misma dirección, distinto portal
    def quitar_portal_nombre(nombre: str):
        return nombre.rsplit('  ', maxsplit=1)[0]

    df_direcciones['Nombre'] = df_direcciones['Nombre'].apply(quitar_portal_nombre)
    df_direcciones.drop_duplicates(subset='Nombre', keep="first", inplace=True)

    # Quitar portal numero
    def quitar_portal_numero(numero: str):
        return numero.split(' ')[0].replace("NUM", "")

    # Conservamos solo los números impares por simpleza del grafo
    df_direcciones['Numero'] = df_direcciones['Numero'].apply(quitar_portal_numero)
    df_direcciones['Numero (entero)'] = pd.to_numeric(df_direcciones['Numero'], errors='coerce')
    df_direcciones = df_direcciones.loc[(df_direcciones['Numero (entero)'] % 2 == 0) |
                                        (df_direcciones['Numero (entero)'].isnull())]
    df_direcciones.drop(columns=['Numero (entero)'], inplace=True)
    return df_cruces, df_direcciones


def crear_dataframe_completo():
    """
    Concatena los dataframes en un uno único, completo y ordenado
    """
    # Concatenar dataframes
    df_completo = pd.concat([dataframe_direcciones, dataframe_cruces], axis=0)

    # Ordenar y rellenar valores incompletos
    df_completo.sort_values(by=['Codigo de via', 'Coordenadas cm'], inplace=True)
    df_completo = df_completo.fillna(method='ffill')
    df_completo.sort_values(by=['Codigo de via', 'Numero'], ignore_index=True, inplace=True)
    return df_completo.drop(columns=['Numero', 'Codigo de cruce', 'Codigo de via cruzada'])


def crear_grafos():
    print("Cargando grafo...")
    # Crear objeto grafo no dirigido
    grafo = Grafo(dirigido=False)

    # Iniciar variables de uso
    posiciones = {}
    distancias = {}
    tiempos = {}
    distritos = {distrito: [] for distrito in dataframe_completo['Nombre del distrito'].unique()}
    color_nodos_dict = {}
    color_aristas = []

    # Iterar sobre las calles del dataframe
    for codigo_calle, dataframe_calle in dataframe_completo.groupby("Codigo de via"):
        # Sacar propiedades de la calle
        tipo_calle = extraer_tipo_calle(dataframe_calle)
        data = {'Codigo': codigo_calle,
                'Distrito': (distrito := dataframe_calle['Nombre del distrito'].mode().values[0]),
                'Tipo': (tipo := tipo_calle if tipo_calle in COLORES_TIPOS else 'OTRO')}
        distritos[distrito] += dataframe_calle['Nombre'].unique().tolist()
        color_calle = COLORES_TIPOS[tipo]
        velocidad_calle = VELOCIDADES_TIPOS[tipo]

        coordenadas_prev = dataframe_calle.iloc[0]['Coordenadas cm']
        nombre_prev = dataframe_calle.iloc[0]['Nombre']
        grafo.agregar_vertice(nombre_prev)

        posiciones[nombre_prev] = coordenadas_prev
        color_nodos_dict[nombre_prev] = color_calle

        for index, fila in dataframe_calle[1:].iterrows():
            # Tomar nombre edificio / portal
            nombre = fila['Nombre']

            if coordenadas_prev != (coordenadas := fila['Coordenadas cm']):
                # Añadimos coordenadas a diccionario de coordenadas
                posiciones[nombre] = coordenadas

                # Añadimos la distancia al diccionario de distancias
                distancias[(nombre_prev, nombre)] = distancia = calcular_distancia(coordenadas_prev, coordenadas)
                tiempos[(nombre_prev, nombre)] = distancia / velocidad_calle

                # Añadimos arista al grafo
                grafo.agregar_arista_vertices(nombre_prev, nombre, data=data, weight=distancia)

                # Añadimos el color de la arista
                color_nodos_dict[nombre] = color_calle
                color_aristas.append(color_calle)

                # Actualizamos nodos previos
                coordenadas_prev = coordenadas
                nombre_prev = nombre

    print("Carga del grafo finalizada con éxito")
    # Convertir grafo a Networkx
    grafo_nx = grafo.convertir_a_NetworkX()

    plt.figure(figsize=(50, 50))
    # Dibujar grafo
    nx.draw(grafo_nx,
            pos=posiciones,
            node_color=(color_nodos := color_nodos_dict.values()),
            edge_color=color_aristas,
            node_size=5)

    # Escribir nombre de cada distrito
    textos = []
    for distrito, nodos in distritos.items():
        if len(nodos) > 1:
            # Hallar centroide
            coords = np.array([posiciones[nodo] for nodo in nodos])

            # Calcular mediana y MAD de las coordenadas
            mediana = np.median(coords, axis=0)
            mad = np.median(np.abs(coords - mediana), axis=0)
            threshold = 2

            # Filtrar coordenadas dentro del threshold
            coords_sin_outlayers = coords[(coords - mediana) / mad < threshold]
            if coords_sin_outlayers.ndim == 1:
                centroide = (coords_sin_outlayers[0], coords_sin_outlayers[1])
            else:
                centroide = tuple(np.mean(coords_sin_outlayers, axis=0))

            # Escribir nombre distrito
            texto = plt.text(centroide[0], centroide[1], distrito,
                             ha='center', va='center', weight='bold',
                             style='italic', fontsize=16)
            textos.append(texto)

    x_centro = [coord_nodo[0] for coord_nodo in posiciones.values()]
    y_centro = [coord_nodo[1] for coord_nodo in posiciones.values()]
    centroide = (sum(x_centro) / len(x_centro), sum(y_centro) / len(y_centro))

    # Escribir "MADRID"
    texto = plt.text(centroide[0], centroide[1] + 100000, "MADRID",
                     ha='center', va='center', weight='bold',
                     fontsize=64)
    textos.append(texto)

    # Guardar y mostrar imagen
    plt.savefig("madrid.png")
    plt.show()
    return grafo, posiciones, color_nodos, color_aristas, textos, {"distancia": distancias, "tiempo": tiempos}


def extraer_tipo_calle(calle: pd.DataFrame):
    """
    Extrae el tipo de una determinada calle. El tipo de
    una calle es la clase de vía más frecuente dentro de
    esta. Si la clase de vía más frecuente es "CRUCE", al
    no representar "CRUCE" ningún tipo de vía, se escogerá
    la siguiente clase más frecuente si existe.
    """
    if len(calle_no_cruces := calle[calle['Clase de la via'] != "CRUCE"]) > 0:
        tipo = calle_no_cruces['Clase de la via'].mode().values[0]
    else:
        tipo = "CRUCE"
    return tipo


def calcular_distancia(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos del grafo.
    """
    radicando = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return radicando ** 0.5


def dibujar_grafo_camino(margen: float = 50000):
    """
    Dibuja el camino mínimo entre los vértices seleccionados.
    """
    # Convertir grafo a networkx
    grafo_madrid_nx = grafo_madrid.convertir_a_NetworkX()

    # Dibujar grafo
    plt.figure(figsize=(50, 50))
    nx.draw(grafo_madrid_nx,
            pos=pos,
            node_color=ncolor,
            edge_color=acolor,
            node_size=200,
            width=3)

    # Resaltar nodos que forman el camino
    nx.draw_networkx_nodes(grafo_madrid_nx,
                           pos=pos,
                           nodelist=camino,
                           node_color='#FF2E2E',
                           node_size=600)
    # Resaltar aristas que forman el camino
    nx.draw_networkx_edges(grafo_madrid_nx,
                           pos=pos,
                           edgelist=[(camino[i], camino[i + 1]) for i in range(len(camino) - 1)],
                           edge_color='#FF2E2E',
                           width=6)
    # Indicar inicio y final
    labels = {origen: 'INICIO', destino: 'FINAL'}
    nx.draw_networkx_labels(grafo_madrid_nx,
                            pos=pos,
                            labels=labels,
                            font_weight='bold',
                            font_size=72)

    # Escribir nombre de cada distrito
    for texto in txt[:-1]:
        posicion = texto.get_position()
        string = texto.get_text()
        plt.text(posicion[0], posicion[1], string,
                 ha='center', va='center', weight='bold',
                 style='italic', fontsize=96)

    # Escribir "MADRID"
    texto = txt[-1]
    posicion = texto.get_position()
    string = texto.get_text()
    plt.text(posicion[0], posicion[1], string,
             ha='center', va='center', weight='bold',
             fontsize=144)

    # Coger posiciones de los nodos que forman el camino
    camino_pos = [pos[nodo] for nodo in camino]

    # Calcular x e y máximas y mínimas para encuadrar imagen
    xmin, ymin = min(camino_pos, key=lambda x: x[0])[0], min(camino_pos, key=lambda x: x[1])[1]
    xmax, ymax = max(camino_pos, key=lambda x: x[0])[0], max(camino_pos, key=lambda x: x[1])[1]

    # Agregar un margen a la imagen
    xmin -= margen
    ymin -= margen
    xmax += margen
    ymax += margen

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Guardar y mostrar imagen
    plt.savefig("camino.png")
    plt.show()


def menu_gps():
    global grafo_madrid, tipo_peso

    tipo_peso_seleccionado = input('¿Qué pretende minimizar, "distancia" o "tiempo"? Seleccione una de estas opciones.'
                                   ' Escriba "exit" para salir del programa.\n')
    if tipo_peso_seleccionado == "exit":
        return None, None, False
    elif tipo_peso_seleccionado not in pesos:
        print('Por favor seleccione una de estas opciones: "distancia" o "tiempo".\n')
        return None, None, True

    if tipo_peso_seleccionado != tipo_peso:
        print("Modificando pesos grafo...")
        grafo_madrid.modificar_aristas(pesos[tipo_peso_seleccionado], atributo="weight")
        tipo_peso = tipo_peso_seleccionado

    source = input('¿Dónde se encuentra ahora mismo?. Responda con el nombre de una calle en MAYÚSCUlAS y su número, '
                   'de forma que queden separado por una coma y un espacio. Por ejemplo: ALBERTO AGUILERA, 25.\n')
    target = input('¿Dónde le interesa ir?. Responda con el nombre de una calle en MAYÚSCUlAS y su número, de forma '
                   'que queden separado por una coma y un espacio. Por ejemplo: GRAN VÍA, 26.\n')

    def encontrar_direccion_cercana(direccion_dada, direcciones):
        calle_direccion, numero_direccion = direccion_dada.split(', ')

        if direccion_dada not in direcciones:
            # Obtener número de la dirección
            numero_direccion = int(numero_direccion)

            # Iniciar variables
            min_diferencia = INFTY
            direccion_cercana = None

            # Iterar sobre direcciones
            for direccion in direcciones:
                numero = int(direccion.split(', ')[1])
                diferencia = abs(numero_direccion - numero)

                if diferencia < min_diferencia:
                    direccion_cercana = direccion
                    min_diferencia = diferencia

                # Debido a que los números de calle son enteros, no puede haber
                # diferencias inferiores a 1.
                if min_diferencia == 1:
                    break

            direccion_dada = direccion_cercana
        return direccion_dada

    if len(via := dataframe_completo[dataframe_completo['Nombre'].str.startswith(source.split(', ')[0])]) > 0:
        source = encontrar_direccion_cercana(source, via['Nombre'])
    else:
        print(f'Lo sentimos, "{source}" no se encuentra en la base de datos.')
        return None, None, True

    if len(via := dataframe_completo[dataframe_completo['Nombre'].str.startswith(target.split(', ')[0])]) > 0:
        target = encontrar_direccion_cercana(target, via['Nombre'])
    else:
        print(f'Lo sentimos, "{target}" no se encuentra en la base de datos.')
        return None, None, True

    return source, target, True


def generar_instrucciones(camino_seleccionado):
    instrucciones = []
    direccion_actual = camino_seleccionado[0]
    direccion_siguiente = camino_seleccionado[1]

    coordenadas_actuales = pos[direccion_actual]
    coordenadas_siguientes = pos[direccion_siguiente]
    
    if coordenadas_siguientes[0] > coordenadas_actuales[0]:
        sentido = "ESTE"
    else:
        sentido = "OESTE"

    def hallar_distancia_arista(source, target):
        datos, peso = grafo_madrid.obtener_arista(source, target)
        return peso * VELOCIDADES_TIPOS[datos['Tipo']] if tipo_peso == "tiempo" else peso

    for i in range(len(camino_seleccionado) - 1):
        direccion_actual = camino_seleccionado[i]
        direccion_siguiente = camino_seleccionado[i + 1]
        calle_actual = direccion_actual.split(', ')[0]
        calle_siguiente = direccion_siguiente.split(', ')[0]

        if calle_actual == calle_siguiente or calle_siguiente == "CRUCE":
            distancia = hallar_distancia_arista(direccion_actual, direccion_siguiente)
            if len(instrucciones) > 0 and instrucciones[-1][0] == "RECTO":
                instrucciones[-1][2] += distancia
            else:
                instrucciones.append(["RECTO", calle_actual, distancia])
        else:
            coordenadas_actuales = pos[direccion_actual]
            coordenadas_siguientes = pos[direccion_siguiente]
            if sentido == "NORTE":
                if coordenadas_siguientes[0] > coordenadas_actuales[0]:
                    instrucciones.append(["DERECHA", calle_siguiente, 0])
                    sentido = "ESTE"
                elif coordenadas_siguientes[0] < coordenadas_actuales[0]:
                    instrucciones.append(["IZQUIERDA", calle_siguiente, 0])
                    sentido = "OESTE"
                    
            elif sentido == "SUR":
                if coordenadas_siguientes[0] < coordenadas_actuales[0]:
                    instrucciones.append(["DERECHA", calle_siguiente, 0])
                    sentido = "OESTE"
                elif coordenadas_siguientes[0] > coordenadas_actuales[0]:
                    instrucciones.append(["IZQUIERDA", calle_siguiente, 0])
                    sentido = "ESTE"

            elif sentido == "ESTE":
                if coordenadas_siguientes[1] < coordenadas_actuales[1]:
                    instrucciones.append(["DERECHA", calle_siguiente, 0])
                    sentido = "SUR"
                elif coordenadas_siguientes[1] > coordenadas_actuales[1]:
                    instrucciones.append(["IZQUIERDA", calle_siguiente, 0])
                    sentido = "NORTE"
                    
            elif sentido == "OESTE":
                if coordenadas_siguientes[1] > coordenadas_actuales[1]:
                    instrucciones.append(["DERECHA", calle_siguiente, 0])
                    sentido = "NORTE"
                elif coordenadas_siguientes[1] < coordenadas_actuales[1]:
                    instrucciones.append(["IZQUIERDA", calle_siguiente, 0])
                    sentido = "SUR"
    return verbalizar_instrucciones(instrucciones)


def verbalizar_instrucciones(lista_instrucciones):
    instrucciones_verbalizadas = []
    for instruccion in lista_instrucciones:
        if instruccion[0] == "RECTO":
            instrucciones_verbalizadas.append(f"Siga recto durante {round(instruccion[2]/100)} metros"
                                              f" por {instruccion[1]}")
        elif instruccion[0] == "DERECHA":
            instrucciones_verbalizadas.append(f"Gire a la derecha por {instruccion[1]}")
        else:
            instrucciones_verbalizadas.append(f"Gire a la izquierda por {instruccion[1]}")
    return instrucciones_verbalizadas


if __name__ == '__main__':
    try:
        # Extraer datos de los dataframes
        dataframe_cruces, dataframe_direcciones = extraer_datos()

        # Concatenar dataframes en uno solo
        dataframe_completo = crear_dataframe_completo()

        # Guardar csv
        dataframe_completo.to_csv('cruces_y_direcciones.csv')

        # Configuración de pesos del grafo
        tipo_peso = PESO_PREDETERMINADO

        # Crear grafo
        grafo_madrid, pos, ncolor, acolor, txt, pesos = crear_grafos()
        running = True

        while running:
            origen, destino, running = menu_gps()

            if origen and destino:
                # Calcular camino mínimo
                try:
                    print(f"Calculando ruta de {origen} a {destino}...")
                    camino = grafo_madrid.camino_minimo(origen, destino)
                    dibujar_grafo_camino(margen=50000)
                    instrucciones_gps = generar_instrucciones(camino)
                    print(instrucciones_gps)
                except ValueError:
                    print(f"{destino} no es alcanzable desde {origen}. Por favor seleccione otro origen u"
                          f" otro destino.")
                print()
    except KeyboardInterrupt:
        print("Salida forzosa del programa. Puede que los datos no se hayan grabado correctamente.")
