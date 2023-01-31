import networkx as nx
import sys

from heapq import heappop, heappush
from itertools import count
from typing import List, Tuple, Dict

from utils import *

INFTY = sys.float_info.max


class Grafo:
    # Diseñar y construir la clase grafo

    def __init__(self, dirigido=False):
        """ Crea un grafo dirigido o no dirigido.
        
        Args:
            dirigido: Flag que indica si el grafo es dirigido o no.
        Returns: Grafo o grafo dirigido (según lo indicado por el flag)
        inicializado sin vértices ni aristas.
        """
        self._dirigido = dirigido
        self._nodos = set()

        if not self._dirigido:
            self._relaciones = {}
        if self._dirigido:
            self._predecesores, self._sucesores = {}, {}

    def __len__(self):
        """ Devuelve el número de nodos que tiene el grafo.

        Returns: Número entero de nodos.
        """
        return len(self._nodos)

    def __contains__(self, v: object):
        """ Indica si el nodo está en el grafo o no.

        Returns: True si está en el grafo, False si no.
        """
        return v in self._nodos

    def __getitem__(self, v: object):
        """ Devuelve los sucesores del nodo v.

        Returns: Diccionario de vecinos del nodo
        """
        if self._dirigido:
            return self._sucesores[v] if v in self._nodos else None
        else:
            return self._relaciones[v] if v in self._nodos else None

    # Operaciones básicas del TAD #
    def es_dirigido(self) -> bool:
        """ Indica si el grafo es dirigido o no.
        
        Args: None
        Returns: True si el grafo es dirigido, False si no.
        """
        return self._dirigido

    def agregar_vertice(self, v: object) -> None:
        """ Agrega el vértice v al grafo. El vértice v no puede ser None.
        
        Args: v vértice que se quiere agregar
        Returns: None
        """
        if v is None:
            return

        if v not in self._nodos:
            if self._dirigido:
                self._sucesores[v], self._predecesores[v] = {}, {}
            else:
                self._relaciones[v] = {}
        self._nodos.add(v)
        return

    def agregar_arista(self, s: object, t: object, data: object, weight: float = 1) -> None:
        """ Si los objetos s y t son vértices del grafo, agrega
        una arista al grafo que va desde el vértice s hasta el vértice t
        y le asocia los datos "data" y el peso weight.
        En caso contrario, no hace nada.
        
        Args:
            s: vértice de origen (source)
            t: vértice de destino (target)
            data: datos de la arista
            weight: peso de la arista
        Returns: None
        """
        if s in self._nodos and t in self._nodos:
            data_dict = {'data': data, 'weight': weight}
            if self._dirigido:
                self._sucesores[s][t] = data_dict
            else:
                self._relaciones[s][t] = self._relaciones[t][s] = data_dict
        return

    def agregar_arista_vertices(self, s: object, t: object, data: object, weight: float = 1) -> None:
        """ Agrega una arista al grafo que va desde el vértice s
        hasta el vértice t, sean o no vértices del grafo, y le asocia
        los datos "data" y el peso weight.
        En caso contrario, no hace nada.

        Args:
            s: vértice de origen (source)
            t: vértice de destino (target)
            data: datos de la arista
            weight: peso de la arista
        Returns: None
        """
        self.agregar_vertice(s)
        self.agregar_vertice(t)
        return self.agregar_arista(s, t, data, weight)

    def eliminar_vertice(self, v: object) -> None:
        """ Si el objeto v es un vértice del grafo lo elimina.
        Si no, no hace nada.
        
        Args: v vértice que se quiere eliminar
        Returns: None
        """
        if v in self._nodos:
            self._nodos.remove(v)
            if self._dirigido:
                del self._sucesores[v]
                for n in self._predecesores[v]:
                    del self._sucesores[n][v]
                del self._predecesores[v]
            else:
                for n in self._relaciones[v]:
                    del self._relaciones[n][v]
                del self._relaciones[v]
        return

    def eliminar_arista(self, s: object, t: object) -> None:
        """ Si los objetos s y t son vértices del grafo y existe
        una arista de u a v la elimina.
        Si no, no hace nada.
        
        Args:
            s: vértice de origen de la arista
            t: vértice de destino de la arista
        Returns: None
        """
        try:
            if not self._dirigido:
                del self._relaciones[s][t], self._relaciones[t][s]
            else:
                del self._sucesores[s][t], self._predecesores[t][s]
        except KeyError:
            pass
        return

    def obtener_arista(self, s: object, t: object) -> Tuple[object, float] or None:
        """ Si los objetos s y t son vértices del grafo y existe
        una arista de s a t, devuelve sus datos y su peso en una tupla.
        Si no, devuelve None
        
        Args:
            s: vértice de origen de la arista
            t: vértice de destino de la arista
        Returns: Una tupla (a, w) con los datos de la arista "a" y su peso
        "w" si la arista existe. None en caso contrario.
        """
        if s in self._nodos and t in self._relaciones:
            try:
                arista = self._relaciones[s][t] if not self._dirigido else self._sucesores[s][t]
                return tuple([value for __, value in arista.items()])
            except KeyError:
                return None

    def obtener_peso(self, s: object, t: object):
        """ Si los objetos s y t son vértices del grafo y existe
        una arista de s a t, devuelve su peso
        Args:
            s: vértice de origen de la arista
            t: vértice de destino de la arista
        Returns: Peso de la arista.
        """
        return self.obtener_arista(s=s, t=t)[1]

    def modificar_aristas(self, diccionario: Dict[Tuple[object, object], object], atributo: str) -> None:
        """
        Esta función modifica los valores de un atributo específico de
        las aristas del grafo por unos dados. Los atributos en concreto
        son: "weight" y "data". Si el atributo especificado es otro, la
        función no modificará nada. Si se quiere modificar el peso,
        "weight", todos los valores deberán ser de tipo "float".

        Ejemplo:
            # Crear el siguiente grafo
            G = Grafo()
            G.agregar_arista_vertices("a", "b", "", 3)
            G.agregar_arista_vertices("b", "c", "", 1)

            # Modificar el peso de la arista ("b", "c") a 2.0
            G.modificar_aristas({("b", "c"): 2.0}, "weight")

            # Imprimir resultado
            print(G.obtener_peso("b", "c"))

        Args:
            diccionario: diccionario cuyas claves son tuplas de la forma
            (origen, destino) y sus valores, los nuevos valores a modificar
            atributo: atributo de las aristas a modificar
        Returns: None
        """
        sucesores = self._relaciones if not self._dirigido else self._sucesores
        if atributo == "weight":
            if all(isinstance(value, float) for value in diccionario.values()):
                for src, dest in diccionario.keys():
                    sucesores[src][dest]['weight'] = diccionario[(src, dest)]
                    if not self._dirigido:
                        sucesores[dest][src]['weight'] = diccionario[(src, dest)]
            else:
                raise ValueError
        elif atributo == "data":
            for src, dest in diccionario.keys():
                sucesores[src][dest]['data'] = diccionario[(src, dest)]
                if not self._dirigido:
                    sucesores[dest][src]['data'] = diccionario[(src, dest)]
        return

    def lista_adyacencia(self, u: object) -> List[object] or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su lista de adyacencia.
        Si no, devuelve None.
        
        Args: u vértice del grafo
        Returns: Una lista [v1,v2,...,vn] de los vértices del grafo
        adyacentes a u si u es un vértice del grafo y None en caso
        contrario
        """
        return list(self._sucesores[u].keys()) if u in self._nodos and self._dirigido else (
               list(self._relaciones[u].keys()) if u in self._nodos and not self._dirigido else (
                None))

    # Grados de vértices #
    def grado_saliente(self, v: object) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado saliente.
        Si no, devuelve None.
        
        Args: u vértice del grafo
        Returns: El grado saliente (int) si el vértice existe y
        None en caso contrario.
        """
        if v in self._nodos:
            if not self._dirigido:
                return len(self._relaciones[v])
            else:
                return len(self._sucesores[v])
        else:
            return None

    def grado_entrante(self, v: object) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado entrante.
        Si no, devuelve None.
        
        Args: u vértice del grafo
        Returns: El grado entrante (int) si el vértice existe y
        None en caso contrario.
        """
        if v in self._nodos:
            if not self._dirigido:
                return len(self._relaciones[v])
            else:
                return len(self._predecesores[v])
        else:
            return None

    def grado(self, v: object) -> int or None:
        """ Si el objeto v es un vértice del grafo, devuelve
        su grado si el grafo no es dirigido y su grado saliente si
        es dirigido.
        Si no pertenece al grafo, devuelve None.
        
        Args: v vértice del grafo
        Returns: El grado (int) o grado saliente (int) según corresponda
        si el vértice existe y None en caso contrario.
        """
        return self.grado_saliente(v)

    def hay_camino(self, origen: object, destino: object) -> bool:
        """ Calcula si hay o no un camino desde el vértice origen hasta el vértice
        destino.

        Args:
            origen: vértice del grafo de origen
            destino: vértice del grafo de destino
        Returns: True si hay camino, False si no.
        """
        try:
            self.camino_minimo(origen, destino)
            return True
        except ValueError:
            return False

    # Algoritmos #
    def dijkstra(self, origen: object) -> Dict[object, object]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo partiendo
        del vértice "origen" usando el algoritmo de Dijkstra. Calcula únicamente
        el árbol de la componente conexa que contiene a "origen".
        
        Args: origen vértice del grafo de origen
        Returns: Devuelve un diccionario que indica, para cada vértice alcanzable
        desde "origen", qué vértice es su padre en el árbol abarcador mínimo.
        """
        return self._dijkstra_caminos(origen)

    def camino_minimo(self, origen: object, destino: object) -> List[object]:
        """ Calcula el camino mínimo desde el vértice origen hasta el vértice
        destino utilizando el algoritmo de Dijkstra.

        Args:
            origen: vértice del grafo de origen
            destino: vértice del grafo de destino
        Returns: Devuelve una lista con los vértices del grafo por los que pasa
        el camino más corto entre el origen y el destino. El primer elemento de
        la lista es origen y el último destino.
        """
        dijkstra_padres = self._dijkstra_caminos(origen, destino)
        camino = []
        vertice = destino

        while vertice != origen:
            camino += [vertice]
            vertice = dijkstra_padres[vertice]
        camino += [origen]

        return camino[::-1]

    # Algoritmos #
    def _dijkstra_caminos(self, origen: object, destino: object = None) -> Dict[object, List[object]]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo partiendo
        del vértice "origen" usando el algoritmo de Dijkstra. Calcula únicamente
        el árbol de la componente conexa que contiene a "origen".

        Args: origen vértice del grafo de origen
        Returns: Devuelve un diccionario que indica, para cada vértice alcanzable
        desde "origen", su camino desde el origen.
        """
        sucesores = self._relaciones if not self._dirigido else self._sucesores
        distancias = {vertice: INFTY for vertice in self._nodos}
        distancias[origen] = 0
        q, visitados = [], {}
        padres = {origen: None}
        if origen not in self._nodos:
            raise ValueError("El origen especificado no forma parte del grafo.")
        heappush(q, (0, next(contador := count()), origen))

        while q:
            distancia_actual, _, vertice_actual = heappop(q)
            if vertice_actual in visitados:
                continue

            visitados[vertice_actual] = True

            if vertice_actual == destino:
                break

            for vecino, metadatos in sucesores[vertice_actual].items():
                distancia = distancias[vertice_actual] + metadatos["weight"]
                if distancias[vecino] > distancia:
                    distancias[vecino] = distancia
                    heappush(q, (distancia, next(contador), vecino))
                    padres[vecino] = vertice_actual
        return padres

    def prim(self) -> Dict[object, object]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.
        
        Args: None
        Returns: Devuelve un diccionario que indica, para cada vértice del
        grafo, qué vértice es su padre en el árbol abarcador mínimo.
        """
        assert not self._dirigido, 'el algoritmo de Prim sólo funciona para grafos no dirigidos.'

        nodos = self._nodos.copy()
        contador = count()
        aam = {}

        while nodos:
            origen = nodos.pop()
            aam[origen] = None
            q = []
            visitados = {origen: True}
            for target, metadatos in self._relaciones[origen].items():
                distancia = metadatos["weight"]
                heappush(q, (distancia, next(contador), origen, target))

            while nodos and q:
                distancia_actual, _, vertice_actual, vertice_target = heappop(q)

                if vertice_target in visitados or vertice_target not in nodos:
                    continue

                aam[vertice_target] = vertice_actual
                visitados[vertice_target] = True
                nodos.discard(vertice_target)

                for vecino, metadatos in self._relaciones[vertice_target].items():
                    if vecino not in visitados:
                        distancia = metadatos["weight"]
                        heappush(q, (distancia, next(contador), vertice_target, vecino))
        return aam

    def kruskal(self) -> List[Tuple[object, object]]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Kruskal.
        
        Args: None
        Returns: Devuelve una lista [(s1,t1),(s2,t2),...,(sn,tn)]
        de los pares de vértices del grafo
        que forman las aristas del arbol abarcador mínimo.
        """
        assert not self._dirigido, 'el algoritmo de Kruskal sólo funciona para grafos no dirigidos.'

        aristas = {}
        for origen in self._relaciones:
            for destino in (vecinos := self._relaciones[origen]):
                if (destino, origen) not in aristas:
                    aristas[(origen, destino)] = vecinos[destino]['weight']
        aristas_ordenadas = sorted(aristas.keys(), key=lambda arista: arista[0])
        del aristas

        subarboles = DisjointSet()
        aristas_aam = []
        for u, v in aristas_ordenadas:
            if subarboles[u] != subarboles[v]:
                aristas_aam.append((u, v))
                subarboles.union(u, v)
        return aristas_aam

    # NetworkX #
    def convertir_a_NetworkX(self) -> nx.Graph or nx.DiGraph:
        """ Construye un grafo o digrafo de Networkx según corresponda
        a partir de los datos del grafo actual.
        
        Args: None
        Returns: Devuelve un objeto Graph de NetworkX si el grafo es
        no dirigido y un objeto DiGraph si es dirigido. En ambos casos,
        los vértices y las aristas son los contenidos en el grafo dado.
        """
        sucesores = self._relaciones if not self._dirigido else self._sucesores
        grafo = nx.Graph() if not self._dirigido else nx.DiGraph()
        for src in self._nodos:
            if not (sucesores_nodo := sucesores[src]):
                grafo.add_node(src)
            else:
                for dest in sucesores_nodo:
                    data, weight = sucesores_nodo[dest].values()
                    grafo.add_edge(src, dest, weight=weight, data=data)
        return grafo


def crear_grafo_camino(numero: int, dirigido: bool = False):
    grafo = Grafo(dirigido=dirigido)
    grafo.agregar_vertice(0)
    for num in range(1, numero):
        grafo.agregar_vertice(num)
        grafo.agregar_arista(s=num, t=num-1, data=None)
    return grafo


def crear_grafo_circular(numero: int, dirigido: bool = False):
    grafo = crear_grafo_camino(numero, dirigido=dirigido)
    grafo.agregar_arista(numero-1, 0, None)
    return grafo


def crear_grafo_completo(numero: int):
    grafo = Grafo(dirigido=False)
    grafo.agregar_vertice(0)
    for num in range(1, numero):
        grafo.agregar_vertice(num)
        for origen in range(0, num):
            grafo.agregar_arista(s=origen, t=num, data=None)
    return grafo
