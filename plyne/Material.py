import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate


class Material:
    def __init__(self, stress: list, strain: list, name: str = "", props: dict = None, color: str = None, flow=None) -> None:
        if not name:
            name = "Unnamed material"
        if not props:
            props = {}
        self.props: dict = props
        self.name: str = name
        self.stress: np.ndarray = stress
        self.strain: np.ndarray = strain
        if not color:
            r = np.random.randint(0, 255)/255
            g = np.random.randint(0, 255)/255
            b = np.random.randint(0, 255)/255
            color = [r, g, b]
        self.color: str = color
        def fdef(x): return False
        self.failure = flow or fdef

        self.ss_curve = interpolate.interp1d(
            self.strain, self.stress, fill_value=0.0, bounds_error=False)

    def show_ss_curve(self) -> matplotlib.lines.Line2D:
        return plt.gca().plot(self.strain, self.stress, label=self.name)[0]

    def give_stress(self, eps):
        return self.ss_curve(eps)


class SimplifiedSteel(Material):

    def __init__(self, E: float, fy: float, fu: float, epsu: float, **kargs) -> None:
        stress: list = [-fu, -fy, 0, fy, fu]
        strain: list = [-epsu, -fy/E, 0, fy/E, epsu]
        name: str = f"Steel {E}"
        Material.__init__(self, stress, strain, name, **kargs)


class WhitneyConcrete(Material):
    def __init__(self, fc: float, **kargs) -> None:
        b1: float = max(0.65, min(0.85, 0.85-0.05/1000*(fc-28000)))
        self.b1: float = b1
        strain: list = [-0.003, -(1-b1)*0.003, -(1-b1)*0.003, 0]
        stress: list = [-fc*0.85, -fc*0.85, 0, 0]
        name: str = f"Whitney {fc}"
        def f(e): return e < -0.003
        Material.__init__(self, stress, strain, name, flow=f, **kargs)


class ConcreteMander(Material):
    def __init__(self, fc: float, **kargs) -> None:
        # TODO Add fc dependency
        strain: list = [-0.005, -0.004438, -0.003355, -0.002219, -
                        0.00174, -0.001083, -0.000222, 0, 0.000132, 0.001447]
        stress: list = [0, -22063.21, -25378.98, -27579.03, -
                        26782.13, -21739.9, -5461.19, 0, 3270.47, 0]
        name: str = f"Concrete {fc}"
        def f(e): return e < -0.3
        Material.__init__(self, stress, strain, name, flow=f, **kargs)


class RebarSteel(Material):

    def __init__(self, grade: int = 60, **kargs) -> None:
        # TODO Add grade dependency
        stress: list = [-260621.85, -620528.21, -551580.63, -482633.05, -413685.47, -
                        413685.47, 0, 413685.47, 413685.47, 482633.05, 551580.63, 620528.21, 260621.85]
        strain: list = [-0.108, -0.09, -0.045556, -0.018889, -0.01, -
                        0.002069, 0, 0.002069, 0.01, 0.018889, 0.045556, 0.09, 0.108]
        name: str = f"Rebar grade {grade}"
        Material.__init__(self, stress, strain, name, **kargs)
