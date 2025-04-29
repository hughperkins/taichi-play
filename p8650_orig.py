# sample code here
import sys
import numpy as np

import taichi as ti
import taichi.math as tm

from matplotlib import cm

# ti.init(arch=ti.cpu, cpu_max_num_threads = 1, debug = True)
ti.init(arch=ti.cuda)

benchmark = 2

@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name # name of the flow case
        ):
        self.name = name
        self.nx = 301
        self.ny = 301 
       
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

        for i, j in self.old_temperature:
            self.old_temperature[i, j] = self.new_temperature[i, j] = 373.15



    @ti.func
    def GetTempPos(self, x, y):
        res = 0.0
        inf_positive = 873.15
        if y <= 0 or y >= self.ny - 1 or x <= 0 or x >= self.nx - 1:
            res = inf_positive
        else:
            res = self.old_temperature[x, y]
        
        return res
            
    
    @ti.func
    def GetTemp(self, x, y, k):
        res = 0.0
        nx, ny = (x + self.e[k][0]) % self.nx, (y + self.e[k][1]) % self.ny
        # if self.is_INB(x, y, k):
        #     bx, by = (x - self.e[k][0]) % self.nx, (y - self.e[k][1]) % self.ny
        #     T_i = 373.15
        #     T2 = self.GetTempPos(bx, by)

        #     u = (0.5 - self.phi[x, y]) / (self.phi[nx, ny] - self.phi[x, y])
        #     res = (2.0 * T_i + (u - 1.0) * T2) / (1 + u)
        # else:
        #     res = self.GetTempPos(nx, ny)
        res = self.GetTempPos(nx, ny)
        
        return res

    
    @ti.kernel
    def UpdateTemp(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            X = 0.05

            laplacian = 0.0
            Tgrad = ti.Vector([0.0, 0.0])

            inv = ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])

            for k in range(1, 9):
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



    @ti.kernel
    def UpdateImage(self):
        for i, j in self.old_temperature:
            temp_color = (self.old_temperature[i, j] - 300) / 500.0
            self.image[i + 2 * self.nx, j] = ti.Vector([temp_color, 0.0, 0.0])

    
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
            gui.set_image(self.image)
            gui.show()
        


if __name__ == '__main__':
    lbm = lbm_solver(
        name = "LBM"
    )
    lbm.solve()