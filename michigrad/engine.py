# Calcado de Micrograd (https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)
import math


class Value:
    """
    Representa a un valor escalar y su gradiente.

    Este valor es el elemento básico de cada neurona
    de la red neuronal y soporta operaciones aritméticas
    y funciones matemáticas necesarias para el
    cálculo del forward y backward pass.
    """

    def __init__(
        self,
        data: float,
        _children: tuple = (),
        _op: str = "",
        name: str = "",
    ) -> None:
        self.data = data
        self.grad = 0
        self.name = name

        # Función que calcula el gradiente durante el backward pass.
        self._backward = lambda: None

        # Conjunto de nodos padres en el grafo.
        self._prev = set(_children)

        # La operación que creó a este nodo. Útil para graphviz.
        self._op = _op

    def __add__(self, other: "Value") -> "Value":
        """
        Define la suma entre dos objetos Value.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            """
            Define el backward pass para la suma.
            """
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Value") -> "Value":
        """
        Define la multiplicación entre dos objetos Value.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            """
            Define el backward pass para la multiplicación.
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: float) -> "Value":
        """
        Define la potenciación de un objeto Value.
        """
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            """
            Define el backward pass para la potenciación.
            """
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Value":
        """
        Define la función ReLU (Rectified Linear Unit).
        """
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            """
            Define el backward pass para la función ReLU.
            """
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> "Value":
        """
        Define la función exponencial.
        """
        x = self.data
        out = Value(math.exp(x), (self,), f"e^{self.data}")

        def _backward() -> None:
            """
            Define el backward pass para la función exponencial.
            """
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        """
        Realiza el backward pass a través del grafo de autograd.
        """
        topo = []
        visited = set()

        def build_topo(v) -> None:
            """
            Construye el orden topológico de todos los nodos en el grafo.
            """
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> "Value":  # -self
        """
        Define la negación de un objeto Value.
        """
        return self * -1

    def __radd__(self, other: "Value") -> "Value":  # other + self
        """
        Define la suma entre un objeto Value y otro objeto.
        """
        return self + other

    def __sub__(self, other: "Value") -> "Value":  # self - other
        """
        Define la resta entre un objeto Value y otro objeto.
        """
        return self + (-other)

    def __rsub__(self, other: "Value") -> "Value":  # other - self
        """
        Define la resta invertida entre un objeto Value y otro objeto.
        """
        return other + (-self)

    def __rmul__(self, other: "Value") -> "Value":  # other * self
        """
        Define la multiplicación entre un objeto Value y otro.
        """
        return self * other

    def __truediv__(self, other: "Value") -> "Value":  # self / other
        """
        Define la división entre un objeto Value y otro.
        """
        return self * other**-1

    def __rtruediv__(self, other: "Value") -> "Value":  # other / self
        """
        Define la división entre un objeto Value y otro.
        """
        return other * self**-1

    def __repr__(self) -> str:
        """
        Representación en string del objeto Value.
        """
        return f"Value(data={self.data}, grad={self.grad}, name={self.name})"
