import random

from michigrad.engine import Value


class Module:
    """
    Contiene dos métodos que todos los componentes principales
    de la red neuronal (la red misma, las capas, las neuronas) implementan.
    """

    def zero_grad(self) -> None:
        """
        Reiniciar los gradientes de todos los parámetros a cero.
        Se hace antes de hacer backpropagation en cada iteración de entrenamiento.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list[Value]:
        """
        Devuelve todos los parámetros, que están representados como
        objetos Value, del módulo.
        """
        return []


class Neuron(Module):
    """
    Representa a una neurona de una capa de la red neuronal.
    """

    def __init__(self, nin: int, nonlin: bool = True) -> None:
        """
        La neurona tiene una lista de pesos y un sesgo (bias).
        Si nonlin es True, la neurona aplica la función de activación ReLU
        a la salida lineal.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        """
        Representa el forward pass de la neurona, es decir, cómo
        calcula su salida a partir de las entradas x.

        La salida es la suma ponderada de las entradas más el sesgo. Si
        nonlin es True, se aplica la función ReLU a la salida.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        """
        Devuelve una lista con todos los parámetros de la neurona.
        """
        return [*self.w, self.b]

    def __repr__(self) -> str:
        """
        Representación en string de la neurona.
        """
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    Representa a una capa de la red neuronal. Puede tener
    una o más neuronas.
    """

    def __init__(self, nin: int, nout: int, **kwargs) -> None:
        """
        La capa tiene una lista de neuronas, cada una con nin entradas y
        nout salidas.
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """
        Representa el forward pass de la capa, es decir, cómo
        calcula su salida a partir de las entradas x.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        """
        Devuelve una lista con todos los parámetros de la capa. Para esto,
        le pide a cada una de sus neuronas que devuelva SUS parámetros.
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        """
        Representación en string de la capa.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    Representa a una red neuronal que puede tener una o más capas.
    """

    def __init__(self, nin: int, nouts: list[int]) -> None:
        """
        La red neuronal tiene una lista de capas.
        """
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """
        Representa el forward pass de la red neuronal, es decir, cómo
        calcula su salida a partir de las entradas x.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        """
        Devuelve una lista con todos los parámetros de la red neuronal.
        Para esto, le pide a cada una de sus capas que devuelva SUS parámetros,
        y cada capa le pide a cada una de sus neuronas que devuelva SUS parámetros.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        """
        Representación en string de la red neuronal.
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
