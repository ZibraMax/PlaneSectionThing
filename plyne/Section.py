import numpy as np
import triangle as tr
from .Material import Material
import matplotlib.pyplot as plt
REBAR_DIAMETERS = {
    "3": 3/8*2.54/100,
    "4": 4/8*2.54/100,
    "5": 5/8*2.54/100,
    "6": 6/8*2.54/100,
    "7": 7/8*2.54/100,
    "8": 8/8*2.54/100,
    "9": 9/8*2.54/100,
    "10": 10/8*2.54/100,
    "11": 11/8*2.54/100,
    "14": 14/8*2.54/100,
    "18": 18/8*2.54/100,
}
REBAR_AREAS = {}
for bar in REBAR_DIAMETERS:
    area = REBAR_DIAMETERS[bar]**2 * np.pi/4
    REBAR_AREAS[bar] = area


class Section:
    def __init__(self, coords: np.ndarray, material: Material, to_mesh: bool = True) -> None:
        self.coords: np.ndarray = coords
        self.material: material = material
        self.area: float = self._calculate_area()
        self.centroid: list = self._calculate_centroid()
        self.to_mesh = to_mesh

    def _calculate_area(self):
        area = 0.0
        coords = [*self.coords]
        coords.append(self.coords[0])
        for i in range(len(coords)-1):
            area += (coords[i][0]) * (coords[i + 1][1])
            area -= (coords[i][1]) * (coords[i + 1][0])
        return area/2

    def _calculate_centroid(self):
        centroideXTotal = 0
        centroideYTotal = 0
        coords = [*self.coords]
        coords.append(self.coords[0])
        for i in range(len(coords)-1):
            centroideXTotal += ((coords[i][0]) + (coords[i + 1][0]))*(
                (coords[i][0])*(coords[i + 1][1]) - (coords[i+1][0])*(coords[i][1]))
            centroideYTotal += ((coords[i][1]) + (coords[i + 1][1]))*(
                (coords[i][0])*(coords[i + 1][1]) - (coords[i+1][0])*(coords[i][1]))

        x = centroideXTotal/(6*self.area)
        y = centroideYTotal/(6*self.area)
        return [x, y]

    def show(self, ax=None) -> None:
        if not ax:
            ax = plt.gca()
        X, Y = self.coords.T.tolist()
        return ax.fill(X+[X[0]], Y+[Y[0]], color=self.material.color, label=self.material.name)


class Fiber(Section):

    def __init__(self, coords: np.ndarray, material: Material) -> None:
        Section.__init__(self, coords, material, to_mesh=False)

    def axial_load(self, strain: callable) -> float:
        p: float = None
        return p

    def T(self, z: np.ndarray) -> np.ndarray:
        x: np.ndarray = None
        return x

    def show(self, ax=None) -> None:
        if not ax:
            ax = plt.gca()
        X, Y = self.coords.T.tolist()
        ax.fill(X+[X[0]], Y+[Y[0]], color=self.material.color)
        ax.plot(X+[X[0]], Y+[Y[0]], color="k")


class Rectangular(Section):
    def __init__(self, b: float, h: float, material: Material, x: float = 0, y: float = 0) -> None:
        coords: np.ndarray = np.array(
            [[x, y], [x+b, y], [x+b, y+h], [x, y+h]])
        Section.__init__(self, coords, material)


class Circular(Section):
    def __init__(self, r: float, material: Material, x: float = 0, y: float = 0, n: float = 10) -> None:
        coords = np.array(self._generate_coords([x, y], r, n=n))
        Section.__init__(self, coords, material)

    def _generate_coords(self, O: list, r: float, sa: float = 0, a: float = np.pi*2, n: int = 10) -> list[float]:
        coords = []
        h = a/n
        for i in range(n+1):
            theta = sa+h*i
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            coords += [[O[0]+x, O[1]+y]]
        theta = a
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        coords += [[O[0]+x, O[1]+y]]
        return coords


class Composite:
    def __init__(self, sections: list[Section]) -> None:
        self.sections: list[Section] = sections
        self.fibers: list[Fiber] = []
        self.cover = None
        self.min_area = None

    def set_cover(self, cover=0.05):
        self.cover = cover

    def mesh(self) -> None:
        self.fibers: list[Fiber] = []
        coords = []
        seg = []
        th = 0
        regions = []
        holes = []
        for mi, sec in enumerate(self.sections):
            segmi = []
            regions.append([*sec.centroid, mi, 0])
            for i in range(len(sec.coords)-1):
                segmi.append([i+th, i+1+th])
            segmi.append([i+1+th, th])
            th += len(sec.coords)
            coords += sec.coords.tolist()
            seg += segmi
            if not sec.to_mesh:
                self.fibers.append(sec)
                holes.append(sec.centroid)
        A = dict(vertices=coords, segments=seg, regions=regions, holes=holes)
        # FIXME Esta mierda de libreria no funciona
        string_triangulation = "pA"
        if self.min_area:
            string_triangulation += f"a{self.min_area/3}"
        print(string_triangulation)
        B = tr.triangulate(A, string_triangulation)
        vertices = B["vertices"]
        materials = B["triangle_attributes"]
        triangles = B["triangles"]
        for idx, t in enumerate(triangles):
            fiber = self.create_fiber(vertices, t, int(materials[idx][0]))
            self.fibers.append(fiber)

    def create_fiber(self, vertices, gdl, material_index):
        coords = vertices[np.ix_(gdl)]
        material = self.sections[material_index].material
        return Fiber(coords, material)

    def moment_curvature(self, plot: bool = True) -> tuple[list[float]]:
        phi: list[float] = []
        M: list[float] = []
        return phi, M

    def interaction_diagram(self, plot: bool = True) -> tuple[list[float]]:
        M: list[float] = []
        P: list[float] = []
        return M, P

    def show(self, mode: str = "sections") -> None:
        fig = plt.figure()
        ax = plt.gca()
        if mode.lower() == "sections":
            for section in self.sections:
                section.show(ax)
        elif mode.lower() == "fibers":
            for fiber in self.fibers:
                fiber.show(ax)
        ax.set_aspect("equal")

    def add_rebar(self, desc: str, h: float, material: Material, direction: str = "horizontal", x: float = None, y: float = None, n: int = 7):
        designation = desc.split("#")[-1]
        if not self.min_area:
            self.min_area = REBAR_AREAS[designation]
        self.min_area = min(self.min_area, REBAR_AREAS[designation])
        n_bars = int(desc.split("#")[0])
        bar_diameter = REBAR_DIAMETERS[designation]
        h -= bar_diameter
        dh = h/(n_bars-1)
        if not x:
            x = self.cover+bar_diameter/2
        if not y:
            y = self.cover+bar_diameter/2

        for _ in range(n_bars):
            _bar = Circular(bar_diameter/2, material, x, y, n)
            bar = Fiber(_bar.coords, material)
            if direction.lower() == "horizontal":
                x += dh
            elif direction.lower() == "vertical":
                y += dh
            else:
                raise Exception(
                    "Only horizontal and vertical directions are allowed")
            self.sections.append(bar)
