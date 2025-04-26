import taichi as ti

ti.init(arch=ti.gpu)

n = 128

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

quad_size = 1.0 / n
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1
ball_radius = 0.3

dt = 0.001

@ti.kernel
def initialize_mass_points():
    for i, j in ti.ndrange(n, n):
        v[i, j] = [0.0, 0.0, 0.0]
        x[i, j] = [i * quad_size, j * quad_size, 0.0]

current_t = 0.0
initialize_mass_points()

ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

spring_offsets = []
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0):
            spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep():
    # for i, j in ti.ndrange(n, n):
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        v[i] -= [0.0, 9.8 * dt, 0.0]

        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                dir = x_ij.normalized()
                current_dist = x_ij.norm()
                rest_length = quad_size * float(i - j).norm()
                spring_force = - spring_Y * (current_dist / rest_length - 1) * dir
                force += spring_force
                damping_force = - dashpot_damping * quad_size * dir * v_ij.dot(dir)
                force += damping_force

        v[i] += force * dt
        v[i] *= ti.exp(-drag_damping * dt)

        offset_from_ball_center = x[i] - ball_center[0]
        if offset_from_ball_center.norm() <= ball_radius:
            normal = offset_from_ball_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal

        x[i] += dt * v[i]

vertices = ti.Vector.field(3, dtype=float, shape=n * n)
indices = ti.field(dtype=int, shape=n * n * 6)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # First triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Second triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

substeps = int(1 / 60 // dt)
print('substeps', substeps)

while window.running:
    if current_t > 1.5:
        initialize_mass_points()
        current_t = 0.0
    
    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
