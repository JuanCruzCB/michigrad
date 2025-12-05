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
        """
        - El argumento 'data' es el valor escalar que representa este objeto Value.
            - Puede ser un número entero o un número de punto flotante.
        - El argumento '_children' es una tupla de objetos Value. Representan
        a los nodos padres en el grafo.
        - El argumento '_op' es una cadena que representa la operación que
        creó a este nodo (+, *, etc). Útil para graphviz.
        - El argumento 'name' es una cadena que representa el nombre
        de este nodo. Útil para graphviz.
        """
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
        - El argumento 'other' es otro objeto Value.
        - Define cómo se suman dos objetos Value:
            - Se crea un nuevo objeto Value cuyo 'data' es la suma
            de los valores de los dos objetos, y cuyos padres son
            los dos objetos que se están sumando. La operación es "+".
            - Se define la función _backward para calcular los gradientes
            durante el backward pass.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, _children=(self, other), _op="+")

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
        - El argumento 'other' es otro objeto Value.
        - Define cómo se multiplican dos objetos Value:
            - Se crea un nuevo objeto Value cuyo 'data' es el producto
            de los valores de los dos objetos, y cuyos padres son
            los dos objetos que se están multiplicando. La operación es "*".
            - Se define la función _backward para calcular los gradientes
            durante el backward pass.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

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
        - El argumento 'other' es un número entero o float.
        - Define cómo se eleva un objeto Value a un entero o float:
            - Se crea un nuevo objeto Value cuyo 'data' es self.data ** other.
            La operación es "**".
            - Se define la función _backward para calcular los gradientes
            durante el backward pass.
        """
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(data=self.data**other, _children=(self,), _op=f"**{other}")

        def _backward() -> None:
            """
            Define el backward pass para la potenciación.
            La derivada de x^n es n*x^(n-1).
            """
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Value":
        """
        - Define la función ReLU (Rectified Linear Unit):
            - Crea un nuevo objeto Value cuyo 'data' es 0 si self.data es
            menor a cero, o self.data en caso contrario.
        """
        out = Value(
            data=0.0 if self.data < 0 else self.data,
            _children=(self,),
            _op="ReLU",
        )

        def _backward() -> None:
            """
            Define el backward pass para la función ReLU.
            La derivada de ReLU es 1 si x > 0, y 0 en caso contrario.
            """
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> "Value":
        """
        - Define la función exponencial:
            - Crea un nuevo objeto Value cuyo 'data' es e elevado a
            self.data.
        """
        x = self.data
        out = Value(data=math.exp(x), _children=(self,), _op=f"e^{self.data}")

        def _backward() -> None:
            """
            Define el backward pass para la función exponencial.
            La derivada de e^x es e^x.
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

        def build_topo(v: Value) -> None:
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
