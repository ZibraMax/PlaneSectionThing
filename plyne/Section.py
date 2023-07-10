from matplotlib.patches import Patch
import numpy as np
import pygmsh
import triangle as tr
from .Material import Material
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        self.material: Material = material
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
        ax.plot(X+[X[0]], Y+[Y[0]], color="k")
        return ax.fill(X+[X[0]], Y+[Y[0]], color=self.material.color)


class Fiber(Section):

    def __init__(self, coords: np.ndarray, material: Material, simplified: bool = False) -> None:
        Section.__init__(self, coords, material, to_mesh=False)
        A0 = 1/3
        A1 = 0.059715871789770
        A2 = 0.797426985353087
        B1 = 0.470142064105115
        B2 = 0.101286507323456
        W0 = 0.1125
        W1 = 0.066197076394253
        W2 = 0.062969590272413
        X = [A0, A1, B1, B1, B2, B2, A2]
        Y = [A0, B1, A1, B1, A2, B2, B2]
        W = [W0, W1, W1, W1, W2, W2, W2]
        self.Z = np.array([X, Y]).T
        self.W = np.array(W)
        self.simplified = simplified
        if not self.simplified:
            self.x = self.T(self.Z.T)
            self.jacs, self.dpz = self.J(self.Z.T)
            self.detjac = np.linalg.det(self.jacs).flatten()

    def axial_load(self, strain: callable) -> float:
        flow = False
        if self.simplified:
            eps = strain(self.centroid[1])
            flow = self.material.failure(eps) or flow
            return flow, self.area*self.material.give_stress(eps)
        p: float = 0.0
        for i in range(len(self.x)):
            w = self.W[i]
            dj = self.detjac[i]
            y = self.x[i][1]
            eps = strain(y)
            flow = self.material.failure(eps) or flow
            sigma = self.material.give_stress(eps)
            p += sigma*w*dj
        return flow, p

    def T(self, z: np.ndarray) -> np.ndarray:
        p = self.psis(z)
        return p@self.coords

    def J(self, z: np.ndarray) -> np.ndarray:
        dpsis = self.dpsis(z).T
        return dpsis @ self.coords, dpsis

    def psis(self, z: np.ndarray) -> np.ndarray:

        return np.array([
            1.0-z[0]-z[1],
            z[0],
            z[1]]).T

    def dpsis(self, z: np.ndarray) -> np.ndarray:

        kernell = (z[0]-z[0])
        return np.array([
            [-1.0*(1+kernell), -1.0*(1+kernell)],
            [1.0*(1+kernell), 0.0*(1+kernell)],
            [0.0*(1+kernell), 1.0*(1+kernell)]
        ])

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
        h = a/(n)
        for i in range(n):
            theta = sa+h*i
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
        self.base = sections[0]
        self.update_materials()

    def set_cover(self, cover=0.05):
        self.cover = cover

    def update_materials(self):
        self.h = 0.0
        maxy = -np.inf
        miny = np.inf
        self.materials: list[Material] = []
        for sec in self.sections:
            maxy = max(max(sec.coords[:, 1]), maxy)
            miny = min(min(sec.coords[:, 1]), miny)

            if not sec.material in self.materials:
                self.materials.append(sec.material)
        self.h = maxy-miny
        self.ymin = miny
        self.cy = self.h/2+miny  # MUST BE COMPOSITE CENTROID!

    def mesh(self, verbose=False):
        self.fibers: list[Fiber] = []

        with pygmsh.geo.Geometry() as geom:
            holes = []
            for i, sec in enumerate(self.sections[1:]):
                coords = sec.coords
                poly = geom.add_polygon(coords.tolist(),
                                        make_surface=sec.to_mesh)
                holes.append(poly.curve_loop)
                if sec.to_mesh:
                    geom.add_physical(poly.surface, f"{i+1}")
                else:
                    self.fibers.append(sec)

            # mesh base with holes
            i = 0
            sec = self.sections[0]
            coords = sec.coords
            poly = geom.add_polygon(coords.tolist(),
                                    holes=holes)
            geom.add_physical(poly.surface, f"{i}")

            mesh = geom.generate_mesh(order=1, verbose=verbose)
        vertices = mesh.points[:, :-1]
        triangles = mesh.cells_dict["triangle"]
        by_materials = mesh.cell_sets_dict
        for mat in by_materials:
            mat_triangles = by_materials[mat]["triangle"]
            for idx in mat_triangles:
                t = triangles[idx]
                fiber = self.create_fiber(vertices, t, int(mat))
                self.fibers.append(fiber)
        return mesh

    def mesh_traingle(self, area=None) -> None:
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
        string_triangulation = "pqA"
        if area:
            string_triangulation += f"a{area}"
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
        PHI: list[float] = [0]
        EPSTOP: list[float] = [0]
        EPSBOTTOM: list[float] = [0]
        dphi = 0.0002
        M: list[float] = [0]
        h = self.h
        c = 0.5*h
        C: list[float] = []
        phi = dphi
        n = 500
        dc = 0.001*h
        for i in tqdm(range(n)):
            cp1 = c+dc
            flow = False
            flag = True
            for j in range(100):

                flowax, P = self.axial_force(c, phi)
                _, Pcp1 = self.axial_force(cp1, phi)

                dpdc = (Pcp1-P)/(cp1-c)
                cp1 = c
                delta = P/dpdc
                c = c - delta

                if abs(delta) < 0.0001:
                    flag = False
                    break
            if flag:
                print("No converge C")

            def strain(y): return (self.h-c-y)*phi
            flowmm, moment = self.moment(c, phi)
            PHI.append(phi)
            M.append(moment)
            if len(C) == 0:
                C.append(c)
            C.append(c)
            EPSTOP.append(strain(self.h))
            EPSBOTTOM.append(strain(0))
            flow = flow or flowax or flowmm
            phi += dphi
            if flow:
                break
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            ax.plot(PHI, M)
            ax.grid()
            ax.set_xlabel("phi")
            ax.set_ylabel("M")

            ax = fig.add_subplot(2, 2, 2)
            ax.plot(PHI, C)
            ax.grid()
            ax.set_xlabel("phi")
            ax.set_ylabel("C")

            ax = fig.add_subplot(2, 2, 3)
            ax.plot(PHI, EPSTOP)
            ax.grid()
            ax.set_xlabel("phi")
            ax.set_ylabel("eps_top")

            ax = fig.add_subplot(2, 2, 4)
            ax.plot(PHI, EPSBOTTOM)
            ax.grid()
            ax.set_xlabel("phi")
            ax.set_ylabel("eps_bottom")

            plt.tight_layout()
            plt.show()
        return PHI, M

    def axial_force(self, c, phi):
        P = 0.0
        flow = False
        def strain(y): return (self.h-c-(y-self.ymin))*phi
        for fiber in self.fibers:
            dflow, p = fiber.axial_load(strain)
            flow = dflow or flow
            P += p
        return flow, P

    def moment(self, c, phi):
        M = 0.0
        flow = False
        def strain(y): return (self.h-c-(y-self.ymin))*phi
        for fiber in self.fibers:
            yc = fiber.centroid[1]
            dflow, p = fiber.axial_load(strain)
            flow = dflow or flow
            d = -(yc - self.cy)
            M += p*d
        return flow, M

    def interaction_diagram(self, ecu=None, etu=None, plot: bool = True, n=100) -> tuple[list[float]]:
        M: list[float] = []
        P: list[float] = []

        if ecu:
            failure = "C"
        elif etu:
            failure = "T"
        else:
            failure = "AUTO"

        ef = -0.003

        C = np.linspace(0.0001, self.h*0.9999, n)
        for c in tqdm(C):
            phi = -ef/c
            flowp, p = self.axial_force(c, phi)
            flowm, m = self.moment(c, phi)
            M.append(m)
            P.append(-p)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            ax.plot(M, P)
            ax.grid()
            ax.set_xlabel("M")
            ax.set_ylabel("P")

            ax = fig.add_subplot(1, 3, 2)
            ax.plot(C, P)
            ax.grid()
            ax.set_xlabel("C")
            ax.set_ylabel("P")

            ax = fig.add_subplot(1, 3, 3)
            ax.plot(C, M)
            ax.grid()
            ax.set_xlabel("C")
            ax.set_ylabel("M")

            plt.tight_layout()
            plt.show()

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
        legend_elements = []
        for mat in self.materials:
            legend_elements.append(Patch(facecolor=mat.color, edgecolor='k',
                                         label=mat.name))
        ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0)

    def add_rebar(self, desc: str, h: float, material: Material, direction: str = "horizontal", x: float = None, y: float = None, n: int = 20):
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
            bar = Fiber(_bar.coords, material, simplified=True)
            if direction.lower() == "horizontal":
                x += dh
            elif direction.lower() == "vertical":
                y += dh
            else:
                raise Exception(
                    "Only horizontal and vertical directions are allowed")
            self.sections.append(bar)
        self.update_materials()
