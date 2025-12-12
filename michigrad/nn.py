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
        Devuelve todos los parámetros del módulo, que
        están representados como objetos Value.
        """
        return []


class Neuron(Module):
    """
    Representa a una neurona de una capa de la red neuronal.
    """

    def __init__(self, nin: int) -> None:
        """
        - El argumento 'nin' indica la cantidad de pesos de la neurona.
        - A su vez, la neurona tiene un único sesgo (bias) que se
        suma a la salida lineal.

        Al crearse, la neurona crea 'nin' pesos, inicializados con
        valores aleatorios entre -1 y 1. Además, inicializa su sesgo en 0.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x: list[Value]) -> Value:
        """
        - Representa el forward pass de la neurona, es decir, cómo
        calcula su salida a partir de las entradas x.
        - El argumento 'x' es una lista de objetos Value que representan
        las entradas a la neurona.
        - La salida de la neurona es un objeto Value que se calcula como
        la suma ponderada de las entradas más el sesgo. Es decir,
        una sumatoria de cada entrada multiplicada por su peso
        correspondiente, y al obtener la suma total, se le suma el sesgo.
        """
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self) -> list[Value]:
        """
        Devuelve una lista de objetos Value que representan a
        todos los parámetros de la neurona: todos sus pesos, y
        además el sesgo.
        """
        return self.w + [self.b]

    def __repr__(self) -> str:
        """
        Representación en string de la neurona.
        """
        return f"{'Linear'}Neuron({len(self.w)})"


class ReLU(Module):
    """
    Representa a una capa de la red neuronal que aplica la función de activación
    ReLU a cada una de sus entradas.
    """

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """
        - Representa el forward pass de la capa, es decir, cómo
        calcula su salida a partir de las entradas x.
        - El argumento 'x' es una lista de objetos Value que representan
        las entradas a la capa.
        - La salida de la capa es una lista de objetos Value, donde
        a cada entrada se le ha aplicado la función ReLU.
        """
        out = [xi.relu() for xi in x]
        return out[0] if len(out) == 1 else out


class Tanh(Module):
    """
    Representa a una capa de la red neuronal que aplica la función de activación
    tangente hiperbólica (tanh) a cada una de sus entradas.
    """

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """
        - Representa el forward pass de la capa, es decir, cómo
        calcula su salida a partir de las entradas x.
        - El argumento 'x' es una lista de objetos Value que representan
        las entradas a la capa.
        - La salida de la capa es una lista de objetos Value, donde
        a cada entrada se le ha aplicado la función tanh.
        """
        out = [xi.tanh() for xi in x]
        return out[0] if len(out) == 1 else out


class Sigmoid(Module):
    """
    Representa a una capa de la red neuronal que aplica la función de activación
    sigmoide a cada una de sus entradas.
    """

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """
        - Representa el forward pass de la capa, es decir, cómo
        calcula su salida a partir de las entradas x.
        - El argumento 'x' es una lista de objetos Value que representan
        las entradas a la capa.
        - La salida de la capa es una lista de objetos Value, donde
        a cada entrada se le ha aplicado la función sigmoide.
        """
        out = [xi.sigmoid() for xi in x]
        return out[0] if len(out) == 1 else out


class Layer(Module):
    """
    Representa a una capa de la red neuronal. Puede tener
    una o más neuronas.
    """

    def __init__(self, nin: int, nout: int) -> None:
        """
        - El argumento 'nin' indica la cantidad de pesos que cada
        neurona de la capa debe tener.
        - El argumento 'nout' indica la cantidad de neuronas que
        debe tener la capa.
        - El argumento '**kwargs' permite pasar argumentos a cada
        neurona al momento de crearlas. En este caso, el único argumento
        adicional posible es 'nonlin', que indica si la neurona debe aplicar
        o no una función de activación no lineal (ReLU) a su salida.

        Al crearse, la capa crea 'nout' neuronas, cada una con 'nin' pesos.
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """
        - Representa el forward pass de la capa, es decir, cómo
        calcula su salida a partir de las entradas x.
        - El argumento 'x' es una lista de objetos Value que representan
        las entradas a la capa.
        - La salida de la capa depende de la cantidad de neuronas que tenga:
          - Si tiene una única neurona, la salida es el resultado de
            pasarle la lista de entradas 'x' a esa neurona, lo cual es
            un objeto Value.
          - Si tiene más de una neurona, la salida es el resultado de
            pasarle la lista de entradas 'x' a cada una de las
            neuronas una por una, lo cual es una lista de objetos Value.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        """
        Devuelve una lista de objetos Value que representan a
        todos los parámetros de la capa. Para esto, le pide a cada
        una de sus neuronas que devuelva SUS parámetros y junta a todos
        esos parámetros en una única lista.
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
        - El argumento 'nin' indica la cantidad de pesos que debe tener
        cada neurona de la PRIMERA capa de la red.
        - El argumento 'nouts' es una lista de enteros que indica
        la cantidad de neuronas que debe tener cada capa de la red.
        - La red neuronal tiene una lista de capas.
        """
        sz = [nin] + nouts
        self.layers = [Layer(nin=sz[i], nout=sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """
        - Representa el forward pass de la red neuronal, es decir, cómo
        calcula su salida a partir de las entradas x.
        - El argumento 'x' es una lista de objetos Value que representan
        las entradas a la red neuronal.
        - La salida de la red neuronal depende de la cantidad de capas que tenga:
          - Si tiene una única capa, la salida es el resultado de pasarle la lista de
            entradas 'x' a esa capa, lo cual es un objeto Value.
          - Si tiene más de una capa, la salida es el resultado de pasarle la lista de
            entradas 'x' a cada una de las capas una por una, lo cual es una lista de
            objetos Value.
        """
        for layer in self.layers:
            z = layer(x)
        return z

    def parameters(self) -> list[Value]:
        """
        Devuelve una lista de objetos Value que representan a
        todos los parámetros de la red neuronal. Para esto, le pide a cada
        una de sus capas que devuelva SUS parámetros, y junta a todos
        esos parámetros en una única lista.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        """
        Representación en string de la red neuronal.
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
