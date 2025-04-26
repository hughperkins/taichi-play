# sample code here
import sys
import numpy as np

import taichi as ti
import taichi.math as tm

from matplotlib import cm

# ti.init(arch=ti.cpu, cpu_max_num_threads = 1, debug = True)
ti.init(arch=ti.cpu)

benchmark = 2

T_hot  = 500.0
T_cold = 300.0

@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name # name of the flow case
        ):
        self.name = name
        self.nx = 301
        self.ny = 301 
       
        # make room for 2 extra rows/cols, one on each side
        self.old_temperature = ti.field(float, shape=(self.nx, self.ny))
        self.new_temperature = ti.field(float, shape=(self.nx, self.ny))

        self.w = ti.field(float, shape=9)
        self.e = ti.Vector.field(2, int, shape=9)

        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(3 * self.nx, self.ny))  # RGB image


    @ti.kernel
    def init(self):
        self.e[0] = ti.Vector([0, 0])
        self.e[1] = ti.Vector([1, 0])
        self.e[2] = ti.Vector([0, 1])
        self.e[3] = ti.Vector([-1, 0])
        self.e[4] = ti.Vector([0, -1])
        self.e[5] = ti.Vector([1, 1])
        self.e[6] = ti.Vector([-1, 1])
        self.e[7] = ti.Vector([-1, -1])
        self.e[8] = ti.Vector([1, -1])

        self.w[0] = 4.0 / 9.0
        self.w[1] = 1.0 / 9.0
        self.w[2] = 1.0 / 9.0
        self.w[3] = 1.0 / 9.0
        self.w[4] = 1.0 / 9.0
        self.w[5] = 1.0 / 36.0
        self.w[6] = 1.0 / 36.0
        self.w[7] = 1.0 / 36.0
        self.w[8] = 1.0 / 36.0

        # for i, j in self.old_temperature:
        #     self.old_temperature[i, j] = self.new_temperature[i, j] = 373.15

        # interior
        for i, j in ti.ndrange(self.nx - 1, self.ny - 1):
            self.old_temperature[i+1, j+1] = 373.15
            self.new_temperature[i+1, j+1] = 373.15

        # left/right walls
        for j in range(self.ny):
            self.old_temperature[0,    j] = T_hot
            self.old_temperature[self.nx - 1, j] = T_hot
        # top/bottom walls
        for i in range(self.nx):
            self.old_temperature[i,    0] = T_hot
            self.old_temperature[i, self.ny - 1] = T_hot

        # mirror into new_temperature halo too
        for i, j in ti.ndrange(self.nx, self.ny):
            self.new_temperature[i, j] = self.old_temperature[i, j]
            
    @ti.func
    def GetTempPos(self, x, y):
        x = (x + self.nx) % self.nx
        y = (y + self.ny) % self.ny
        return self.old_temperature[x, y]            
    
    # @ti.func
    # def GetTempPos(self, x, y):
    #     temp = self.old_temperature[x, y]
    #     if x < 0 or x >= self.nx:
    #         temp = T_hot
    #     elif y < 0 or y >= self.ny:
    #         temp = T_cold
    #     return temp
    
    # @ti.func
    # def GetTemp(self, x, y, k):
    #     return self.GetTempPos(x + self.e[k][0], y + self.e[k][1])
    
    @ti.func
    def GetTemp(self, i, j, k):
        dx, dy = self.e[k]
        # i,j here will be in [1..nx], so i+dx in [0..nx+1] is always valid
        return self.old_temperature[i + dx, j + dy]

    @ti.kernel
    def UpdateTemp(self):
        for _i, _j in ti.ndrange(self.nx - 1, self.ny - 1):
            i = _i + 1  # shift into the halo’d array
            j = _j + 1

            X = 0.05

            laplacian = 0.0
            Tgrad = ti.Vector([0.0, 0.0])

            inv = ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])

            for k in range(9):
                new_grad = 1.5 * self.w[k] * (self.GetTemp(i, j, k) - self.GetTemp(i, j, inv[k])) * self.e[k]
                new_lap = 3.0 * self.w[k] * (self.GetTemp(i, j, k) + self.GetTemp(i, j, inv[k]) - 2.0 * self.GetTempPos(i, j))
                Tgrad += new_grad
                laplacian += new_lap

                if (i == 0 and j == 8) or \
                    (i == 300 and j == 8) or \
                    (i == 0 and j == 292) or \
                    (i == 300 and j == 292):

                    print("break", i, j, k, new_lap, laplacian, new_grad, Tgrad)

            self.new_temperature[i, j] = self.old_temperature[i, j] + X * laplacian
            
        for i, j in ti.ndrange(self.nx, self.ny):
            self.old_temperature[i, j] = self.new_temperature[i, j]



    # @ti.kernel
    # def UpdateImage(self):
    #     for i, j in self.old_temperature:
    #         temp_color = (self.old_temperature[i, j] - 300) / 500.0
    #         self.image[i + 2 * self.nx, j] = ti.Vector([temp_color, 0.0, 0.0])
    # @ti.kernel
    # def UpdateImage(self):
    #     for i, j in self.old_temperature:
    #         # get original and its 180°–rotated counterpart
    #         T  = self.old_temperature[i, j]
    #         Tm = self.old_temperature[self.nx - 1 - i, self.ny - 1 - j]
    #         diff = abs(T - Tm)

    #         # normalize / amplify for display
    #         c0 = (T  - 300.0) / 500.0        # red = original
    #         c1 = (Tm - 300.0) / 500.0        # green = mirrored
    #         c2 = diff * 10.0                 # blue = difference

    #         # tile three columns: [orig | mirror | diff]
    #         self.image[i,             j] = ti.Vector([c0, 0.0, 0.0])
    #         self.image[i +    self.nx, j] = ti.Vector([0.0, c1, 0.0])
    #         self.image[i + 2*self.nx, j] = ti.Vector([0.0, 0.0, c2])
    @ti.kernel
    def UpdateImage(self):
        for i, j in self.old_temperature:
            T   = self.old_temperature[i, j]
            Tm  = self.old_temperature[self.nx - 1 - i, self.ny - 1 - j]
            diff = abs(T - Tm)

            # red=original, green=mirror
            c0 = (T  - 300.0) / 500.0
            c1 = (Tm - 300.0) / 500.0

            # if diff>0 → white; else black
            c2 = ti.select(diff > 1e-12, 1.0, 0.0)

            self.image[i,             j] = ti.Vector([c0, 0.0, 0.0])
            self.image[i +    self.nx, j] = ti.Vector([0.0, c1, 0.0])
            self.image[i + 2*self.nx, j] = ti.Vector([c2, c2, c2])
    
    def dump_sym_errors(self, tol=1e-12):
        import numpy as np
        A = self.old_temperature.to_numpy()            # shape (ny, nx)
        M = np.abs(A - A[::-1, ::-1])                  # 180° flipped difference
        ys, xs = np.where(M > tol)
        print(f"found {len(xs)} symmetry errors > {tol:e}")
        for y, x in zip(ys, xs):
            T  = A[y, x]
            Tm = A[-1-y, -1-x]
            d  = M[y, x]
            print(f"({x:3d},{y:3d})  T={T:.6f}  Tm={Tm:.6f}  diff={d:.3e}")

    @ti.kernel
    def CheckSym(self):
        check_temp = True

        for i0, j0 in self.old_temperature:
            i1, j1 = 300 - i0, 300 - j0

            if self.old_temperature[i0, j0]!=self.old_temperature[i0, j1] or \
                self.old_temperature[i0, j0]!=self.old_temperature[i1, j1] or \
                self.old_temperature[i0, j0]!=self.old_temperature[i1, j0]:
                check_temp = False
                print("diff temp, ", i0, j0)

        print("check sym ", check_temp)
    

    def solve(self):
        gui = ti.GUI(self.name, (3 * self.nx, self.ny))
        self.init()
        frame = 0

        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT) and frame < 100:
            sys.stdout.flush()
            frame += 1
            for substep in range(1):
                self.UpdateTemp()
                self.CheckSym()

            
            self.UpdateImage()
            self.dump_sym_errors()
            gui.set_image(self.image)
            gui.show()
        


if __name__ == '__main__':
    lbm = lbm_solver(
        name = "LBM"
    )
    lbm.solve()
