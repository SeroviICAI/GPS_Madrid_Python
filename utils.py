class DisjointSet(object):
    def __init__(self):
        self.parents = {}
        self.pesos = {}

    def __getitem__(self, item: object):
        return self.buscar(item)

    def buscar(self, item: object):
        if item not in self.parents:
            self.parents[item] = item
            self.pesos[item] = 1
            return item

        raiz, camino = self.parents[item], [item]
        while raiz != camino[-1]:
            camino.append(raiz)
            raiz = self.parents[raiz]
        for ancestor in camino:
            self.parents[ancestor] = raiz
        return raiz

    def union(self, x: object, y: object):
        x_raiz = self.buscar(x)
        y_raiz = self.buscar(y)
        if x_raiz == y_raiz:
            return

        if self.pesos[x_raiz] < self.pesos[y_raiz]:
            self.parents[x_raiz] = y_raiz
        elif self.pesos[x_raiz] > self.pesos[y_raiz]:
            self.parents[y_raiz] = x_raiz
        else:
            self.parents[y_raiz] = x_raiz
            self.pesos[x_raiz] += 1
