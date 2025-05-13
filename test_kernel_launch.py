import numpy as np
import taichi as ti
import genesis as gs

ti.init(arch=ti.cpu, offline_cache=False)

@ti.kernel
def foo():
    ...
print("after foo declaration")

@ti.kernel
def bar():
    ...
print("after bar declaration")

# class PARA_LEVEL:
#     ALL = 8

# class gs:
#     ti_int = ti.types.int32
#     ti_vec3 = ti.types.vector(3, dtype=ti.f32)
#     ti_vec7 = ti.types.vector(7, dtype=ti.f32)
#     ti_vec4 = ti.types.vector(4, dtype=ti.f32)
#     ti_vec6 = ti.types.vector(6, dtype=ti.f32)
#     ti_vec2 = ti.types.vector(2, dtype=ti.f32)
#     ti_float = ti.types.float32
#     ti_mat3 = ti.types.matrix(3, 3, dtype=ti.f32)
#     np_float = np.float32
#     PARA_LEVEL = PARA_LEVEL()

#     @staticmethod
#     def _batch_shape(n):
#         return (n,)



# @ti.data_oriented
# class RigidSolver_:
#     def __init__(self):
#         struct_joint_info = ti.types.struct(
#             type=gs.ti_int,
#             sol_params=gs.ti_vec7,
#             q_start=gs.ti_int,
#             dof_start=gs.ti_int,
#             q_end=gs.ti_int,
#             dof_end=gs.ti_int,
#             n_dofs=gs.ti_int,
#             pos=gs.ti_vec3,
#         )
#         self._entities = []
#         self._para_level = 4
#         self.n_joints = 10
#         self._B = 4
#         self.n_links = 5
#         self.n_entities = 3
#         joints_info_shape = self._batch_shape(self.n_joints)
#         self.joints_info = struct_joint_info.field(shape=joints_info_shape, needs_grad=False, layout=ti.Layout.SOA)
#         self._use_hibernation = False

#         self.links = []
#         self.joints = []

#         self._init_entity_fields()
#         self._init_link_fields()

#     def _init_link_fields(self):
#         if self._use_hibernation:
#             self.n_awake_links = ti.field(dtype=gs.ti_int, shape=self._B)
#             self.awake_links = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_links_))

#         struct_link_info = ti.types.struct(
#             parent_idx=gs.ti_int,
#             root_idx=gs.ti_int,
#             q_start=gs.ti_int,
#             dof_start=gs.ti_int,
#             joint_start=gs.ti_int,
#             q_end=gs.ti_int,
#             dof_end=gs.ti_int,
#             joint_end=gs.ti_int,
#             n_dofs=gs.ti_int,
#             pos=gs.ti_vec3,
#             quat=gs.ti_vec4,
#             invweight=gs.ti_vec2,
#             is_fixed=gs.ti_int,
#             inertial_pos=gs.ti_vec3,
#             inertial_quat=gs.ti_vec4,
#             inertial_i=gs.ti_mat3,
#             inertial_mass=gs.ti_float,
#             entity_idx=gs.ti_int,  # entity.idx_in_solver
#         )

#         struct_joint_info = ti.types.struct(
#             type=gs.ti_int,
#             sol_params=gs.ti_vec7,
#             q_start=gs.ti_int,
#             dof_start=gs.ti_int,
#             q_end=gs.ti_int,
#             dof_end=gs.ti_int,
#             n_dofs=gs.ti_int,
#             pos=gs.ti_vec3,
#         )

#         struct_link_state = ti.types.struct(
#             cinr_inertial=gs.ti_mat3,
#             cinr_pos=gs.ti_vec3,
#             cinr_quat=gs.ti_vec4,
#             cinr_mass=gs.ti_float,
#             crb_inertial=gs.ti_mat3,
#             crb_pos=gs.ti_vec3,
#             crb_quat=gs.ti_vec4,
#             crb_mass=gs.ti_float,
#             cdd_vel=gs.ti_vec3,
#             cdd_ang=gs.ti_vec3,
#             pos=gs.ti_vec3,
#             quat=gs.ti_vec4,
#             ang=gs.ti_vec3,
#             vel=gs.ti_vec3,
#             i_pos=gs.ti_vec3,
#             i_quat=gs.ti_vec4,
#             j_pos=gs.ti_vec3,
#             j_quat=gs.ti_vec4,
#             j_vel=gs.ti_vec3,
#             j_ang=gs.ti_vec3,
#             # cd
#             cd_ang=gs.ti_vec3,
#             cd_vel=gs.ti_vec3,
#             root_COM=gs.ti_vec3,
#             mass_sum=gs.ti_float,
#             COM=gs.ti_vec3,
#             mass_shift=gs.ti_float,
#             i_pos_shift=gs.ti_vec3,
#             # cfrc_flat
#             cfrc_flat_ang=gs.ti_vec3,
#             cfrc_flat_vel=gs.ti_vec3,
#             # COM-based external force
#             cfrc_ext_ang=gs.ti_vec3,
#             cfrc_ext_vel=gs.ti_vec3,
#             # net force from external contacts
#             contact_force=gs.ti_vec3,
#             # Flag for links that converge into a static state (hibernation)
#             hibernated=gs.ti_int,
#         )

#         links_info_shape = self._batch_shape(self.n_links)
#         # links_info_shape = self._batch_shape(self.n_links) if self._options.batch_links_info else self.n_links
#         self.links_info = struct_link_info.field(shape=links_info_shape, needs_grad=False, layout=ti.Layout.SOA)
#         self.links_state = struct_link_state.field(
#             shape=self._batch_shape(self.n_links), needs_grad=False, layout=ti.Layout.SOA
#         )

#         links = self.links
#         # self._kernel_init_link_fields(
#         #     links_parent_idx=np.array([link.parent_idx for link in links], dtype=gs.np_int),
#         #     links_root_idx=np.array([link.root_idx for link in links], dtype=gs.np_int),
#         #     links_q_start=np.array([link.q_start for link in links], dtype=gs.np_int),
#         #     links_dof_start=np.array([link.dof_start for link in links], dtype=gs.np_int),
#         #     links_joint_start=np.array([link.joint_start for link in links], dtype=gs.np_int),
#         #     links_q_end=np.array([link.q_end for link in links], dtype=gs.np_int),
#         #     links_dof_end=np.array([link.dof_end for link in links], dtype=gs.np_int),
#         #     links_joint_end=np.array([link.joint_end for link in links], dtype=gs.np_int),
#         #     links_invweight=np.array([link.invweight for link in links], dtype=gs.np_float),
#         #     links_is_fixed=np.array([link.is_fixed for link in links], dtype=gs.np_int),
#         #     links_pos=np.array([link.pos for link in links], dtype=gs.np_float),
#         #     links_quat=np.array([link.quat for link in links], dtype=gs.np_float),
#         #     links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
#         #     links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
#         #     links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
#         #     links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
#         #     links_entity_idx=np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
#         # )

#         # joints_info_shape = self._batch_shape(self.n_joints) if self._options.batch_joints_info else self.n_joints
#         joints_info_shape = self._batch_shape(self.n_joints)
#         self.joints_info = struct_joint_info.field(shape=joints_info_shape, needs_grad=False, layout=ti.Layout.SOA)

#         struct_joint_state = ti.types.struct(
#             xanchor=gs.ti_vec3,
#             xaxis=gs.ti_vec3,
#         )

#         self.joints_state = struct_joint_state.field(
#             shape=self._batch_shape(self.n_joints), needs_grad=False, layout=ti.Layout.SOA
#         )

#         # Make sure that the constraints parameters are valid
#         joints = self.joints
#         joints_sol_params = np.concatenate([joint.sol_params for joint in joints], dtype=gs.np_float)
#         joints_sol_params = _sanitize_sol_params(
#             joints_sol_params, self._sol_constraint_min_resolve_time, self._sol_constraint_resolve_time
#         )

#         self._kernel_init_joint_fields(
#             joints_type=np.array([joint.type for joint in joints], dtype=gs.np_int),
#             joints_sol_params=joints_sol_params,
#             joints_q_start=np.array([joint.q_start for joint in joints], dtype=gs.np_int),
#             joints_dof_start=np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
#             joints_q_end=np.array([joint.q_end for joint in joints], dtype=gs.np_int),
#             joints_dof_end=np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
#             joints_pos=np.array([joint.pos for joint in joints], dtype=gs.np_float),
#         )

#         self.qpos0 = ti.field(dtype=gs.ti_float, shape=self._batch_shape(self.n_qs_))
#         if self.n_qs > 0:
#             init_qpos = self._batch_array(self.init_qpos.astype(gs.np_float))
#             self.qpos0.from_numpy(init_qpos)

#         # Check if the initial configuration is out-of-bounds
#         self.qpos = ti.field(dtype=gs.ti_float, shape=self._batch_shape(self.n_qs_))
#         is_init_qpos_out_of_bounds = False
#         if self.n_qs > 0:
#             init_qpos = self._batch_array(self.init_qpos.astype(gs.np_float))
#             for joint in joints:
#                 if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
#                     is_init_qpos_out_of_bounds |= (joint.dofs_limit[0, 0] > init_qpos[joint.q_start]).any()
#                     is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > joint.dofs_limit[0, 1]).any()
#                     # init_qpos[joint.q_start] = np.clip(init_qpos[joint.q_start], *joint.dofs_limit[0])
#             self.qpos.from_numpy(init_qpos)
#         if is_init_qpos_out_of_bounds:
#             gs.logger.warning(
#                 "Reference robot position exceeds joint limits."
#                 # "Clipping initial position too make sure it is valid."
#             )

#         # This is for IK use only
#         # TODO: support IK with parallel envs
#         self.links_T = ti.Matrix.field(n=4, m=4, dtype=gs.ti_float, shape=self.n_links)

#     def _batch_shape(self, shape=None, first_dim=False, B=None):
#         if B is None:
#             B = self._B

#         if shape is None:
#             return (B,)
#         elif type(shape) in [list, tuple]:
#             return (B,) + shape if first_dim else shape + (B,)
#         else:
#             return (B, shape) if first_dim else (shape, B)

#     def _init_entity_fields(self):
#         if self._use_hibernation:
#             self.n_awake_entities = ti.field(dtype=gs.ti_int, shape=self._B)
#             self.awake_entities = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_entities_))

#         struct_entity_info = ti.types.struct(
#             dof_start=gs.ti_int,
#             dof_end=gs.ti_int,
#             n_dofs=gs.ti_int,
#             link_start=gs.ti_int,
#             link_end=gs.ti_int,
#             n_links=gs.ti_int,
#             geom_start=gs.ti_int,
#             geom_end=gs.ti_int,
#             n_geoms=gs.ti_int,
#             gravity_compensation=gs.ti_float,
#         )

#         struct_entity_state = ti.types.struct(
#             hibernated=gs.ti_int,
#         )

#         self.entities_info = struct_entity_info.field(shape=self.n_entities, needs_grad=False, layout=ti.Layout.SOA)
#         self.entities_state = struct_entity_state.field(
#             shape=self._batch_shape(self.n_entities), needs_grad=False, layout=ti.Layout.SOA
#         )

#         entities = self._entities
#         # self._kernel_init_entity_fields(
#         #     entities_dof_start=np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
#         #     entities_dof_end=np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
#         #     entities_link_start=np.array([entity.link_start for entity in entities], dtype=gs.np_int),
#         #     entities_link_end=np.array([entity.link_end for entity in entities], dtype=gs.np_int),
#         #     entities_geom_start=np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
#         #     entities_geom_end=np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
#         #     entities_gravity_compensation=np.array(
#         #         [entity.gravity_compensation for entity in entities], dtype=gs.np_float
#         #     ),
#         # )

#     @ti.kernel
#     def _kernel_init_joint_fields(
#         self,
#         joints_type: ti.types.ndarray(),
#         joints_sol_params: ti.types.ndarray(),
#         joints_q_start: ti.types.ndarray(),
#         joints_dof_start: ti.types.ndarray(),
#         joints_q_end: ti.types.ndarray(),
#         joints_dof_end: ti.types.ndarray(),
#         joints_pos: ti.types.ndarray(),
#     ):
#         ti.loop_config(serialize=False)
#         for I in ti.grouped(self.joints_info):
#             i = I[0]

#             self.joints_info[I].type = joints_type[i]
#             self.joints_info[I].q_start = joints_q_start[i]
#             self.joints_info[I].dof_start = joints_dof_start[i]
#             self.joints_info[I].q_end = joints_q_end[i]
#             self.joints_info[I].dof_end = joints_dof_end[i]
#             self.joints_info[I].n_dofs = joints_dof_end[i] - joints_dof_start[i]

#             for j in ti.static(range(7)):
#                 self.joints_info[I].sol_params[j] = joints_sol_params[i, j]
#             for j in ti.static(range(3)):
#                 self.joints_info[I].pos[j] = joints_pos[i, j]


#     @ti.kernel
#     def _kernel_forward_kinematics_links_geoms(self, envs_idx: ti.types.ndarray()):
#         ti.loop_config(serialize=False)
#         for i_b in envs_idx:
#             self._func_forward_kinematics(i_b)
#             self._func_transform_COM(i_b)
#             self._func_forward_velocity(i_b)
#             self._func_update_geoms(i_b)

#     @ti.func
#     def _func_transform_COM(self, i_b):
#         self._func_COM_links(i_b)
#         # TODO: Already computed in forward dynamics, but maybe here is a better place
#         # self._func_COM_cd(i_b)
#         # self._func_COM_cdofd(i_b)

#     @ti.func
#     def _func_forward_kinematics(self, i_b):
#         if ti.static(self._use_hibernation):
#             ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
#             for i_e_ in range(self.n_awake_entities[i_b]):
#                 i_e = self.awake_entities[i_e_, i_b]
#                 self._func_forward_kinematics_entity(i_e, i_b)
#         else:
#             ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
#             for i_e in range(self.n_entities):
#                 self._func_forward_kinematics_entity(i_e, i_b)

#     @ti.func
#     def _func_forward_velocity(self, i_b):
#         if ti.static(self._use_hibernation):
#             ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
#             for i_e_ in range(self.n_awake_entities[i_b]):
#                 i_e = self.awake_entities[i_e_, i_b]
#                 self._func_forward_velocity_entity(i_e, i_b)
#         else:
#             ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
#             for i_e in range(self.n_entities):
#                 self._func_forward_velocity_entity(i_e, i_b)

#     @ti.func
#     def _func_forward_kinematics_entity(self, i_e, i_b):
#         for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
#             I_l = [i_l, i_b]
#             # I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
#             l_info = self.links_info[I_l]

#             pos = l_info.pos
#             quat = l_info.quat
#             if l_info.parent_idx != -1:
#                 parent_pos = self.links_state[l_info.parent_idx, i_b].pos
#                 parent_quat = self.links_state[l_info.parent_idx, i_b].quat
#                 pos = parent_pos + gu.ti_transform_by_quat(pos, parent_quat)
#                 quat = gu.ti_transform_quat_by_quat(quat, parent_quat)

#             for i_j in range(l_info.joint_start, l_info.joint_end):
#                 I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
#                 j_info = self.joints_info[I_j]
#                 joint_type = j_info.type
#                 q_start = j_info.q_start
#                 dof_start = j_info.dof_start
#                 I_d = [dof_start, i_b] if ti.static(self._options.batch_dofs_info) else dof_start

#                 # compute axis and anchor
#                 if joint_type == gs.JOINT_TYPE.FREE:
#                     self.joints_state[i_j, i_b].xanchor = ti.Vector(
#                         [self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]]
#                     )
#                     self.joints_state[i_j, i_b].xaxis = ti.Vector([0.0, 0.0, 1.0])
#                 elif joint_type == gs.JOINT_TYPE.FIXED:
#                     pass
#                 else:
#                     axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
#                     if joint_type == gs.JOINT_TYPE.REVOLUTE:
#                         axis = self.dofs_info[I_d].motion_ang
#                     elif joint_type == gs.JOINT_TYPE.PRISMATIC:
#                         axis = self.dofs_info[I_d].motion_vel

#                     self.joints_state[i_j, i_b].xanchor = gu.ti_transform_by_quat(j_info.pos, quat) + pos
#                     self.joints_state[i_j, i_b].xaxis = gu.ti_transform_by_quat(axis, quat)

#                 if joint_type == gs.JOINT_TYPE.FREE:
#                     pos = ti.Vector(
#                         [self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]],
#                         dt=gs.ti_float,
#                     )
#                     quat = ti.Vector(
#                         [
#                             self.qpos[q_start + 3, i_b],
#                             self.qpos[q_start + 4, i_b],
#                             self.qpos[q_start + 5, i_b],
#                             self.qpos[q_start + 6, i_b],
#                         ],
#                         dt=gs.ti_float,
#                     )
#                     quat = gu.ti_normalize(quat)
#                     xyz = gu.ti_quat_to_xyz(quat)
#                     for i in range(3):
#                         self.dofs_state[dof_start + i, i_b].pos = pos[i]
#                         self.dofs_state[dof_start + 3 + i, i_b].pos = xyz[i]
#                 elif joint_type == gs.JOINT_TYPE.FIXED:
#                     pass
#                 elif joint_type == gs.JOINT_TYPE.SPHERICAL:
#                     qloc = ti.Vector(
#                         [
#                             self.qpos[q_start, i_b],
#                             self.qpos[q_start + 1, i_b],
#                             self.qpos[q_start + 2, i_b],
#                             self.qpos[q_start + 3, i_b],
#                         ],
#                         dt=gs.ti_float,
#                     )
#                     xyz = gu.ti_quat_to_xyz(qloc)
#                     for i in range(3):
#                         self.dofs_state[dof_start + i, i_b].pos = xyz[i]
#                     quat = gu.ti_transform_quat_by_quat(qloc, quat)
#                     pos = self.joints_state[i_j, i_b].xanchor - gu.ti_transform_by_quat(j_info.pos, quat)
#                 elif joint_type == gs.JOINT_TYPE.REVOLUTE:
#                     axis = self.dofs_info[I_d].motion_ang
#                     self.dofs_state[dof_start, i_b].pos = self.qpos[q_start, i_b] - self.qpos0[q_start, i_b]
#                     qloc = gu.ti_rotvec_to_quat(axis * self.dofs_state[dof_start, i_b].pos)
#                     quat = gu.ti_transform_quat_by_quat(qloc, quat)
#                     pos = self.joints_state[i_j, i_b].xanchor - gu.ti_transform_by_quat(j_info.pos, quat)
#                 else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
#                     self.dofs_state[dof_start, i_b].pos = self.qpos[q_start, i_b] - self.qpos0[q_start, i_b]
#                     pos = pos + self.joints_state[i_j, i_b].xaxis * self.dofs_state[dof_start, i_b].pos

#             # Skip link pose update for fixed root links to allow the user for manually overwriting them
#             if not (l_info.is_fixed and l_info.parent_idx == -1):
#                 self.links_state[i_l, i_b].pos = pos
#                 self.links_state[i_l, i_b].quat = quat

#     @ti.func
#     def _func_forward_velocity_entity(self, i_e, i_b):
#         for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
#             I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
#             l_info = self.links_info[I_l]

#             cvel_vel = ti.Vector.zero(gs.ti_float, 3)
#             cvel_ang = ti.Vector.zero(gs.ti_float, 3)
#             if l_info.parent_idx != -1:
#                 cvel_vel = self.links_state[l_info.parent_idx, i_b].cd_vel
#                 cvel_ang = self.links_state[l_info.parent_idx, i_b].cd_ang

#             for i_j in range(l_info.joint_start, l_info.joint_end):
#                 I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
#                 j_info = self.joints_info[I_j]
#                 joint_type = j_info.type
#                 q_start = j_info.q_start
#                 dof_start = j_info.dof_start

#                 if joint_type == gs.JOINT_TYPE.FREE:
#                     ## TODO: cdof_dots and cdof_ang_dot
#                     for i_3 in range(3):
#                         cvel_vel = (
#                             cvel_vel
#                             + self.dofs_state[dof_start + i_3, i_b].cdof_vel * self.dofs_state[dof_start + i_3, i_b].vel
#                         )
#                         cvel_ang = (
#                             cvel_ang
#                             + self.dofs_state[dof_start + i_3, i_b].cdof_ang * self.dofs_state[dof_start + i_3, i_b].vel
#                         )

#                     for i_3 in range(3):
#                         (
#                             self.dofs_state[dof_start + i_3, i_b].cdofd_ang,
#                             self.dofs_state[dof_start + i_3, i_b].cdofd_vel,
#                         ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

#                         (
#                             self.dofs_state[dof_start + i_3 + 3, i_b].cdofd_ang,
#                             self.dofs_state[dof_start + i_3 + 3, i_b].cdofd_vel,
#                         ) = gu.motion_cross_motion(
#                             cvel_ang,
#                             cvel_vel,
#                             self.dofs_state[dof_start + i_3 + 3, i_b].cdof_ang,
#                             self.dofs_state[dof_start + i_3 + 3, i_b].cdof_vel,
#                         )

#                     for i_3 in range(3):
#                         cvel_vel = (
#                             cvel_vel
#                             + self.dofs_state[dof_start + i_3 + 3, i_b].cdof_vel
#                             * self.dofs_state[dof_start + i_3 + 3, i_b].vel
#                         )
#                         cvel_ang = (
#                             cvel_ang
#                             + self.dofs_state[dof_start + i_3 + 3, i_b].cdof_ang
#                             * self.dofs_state[dof_start + i_3 + 3, i_b].vel
#                         )

#                 else:
#                     for i_d in range(dof_start, j_info.dof_end):
#                         self.dofs_state[i_d, i_b].cdofd_ang, self.dofs_state[i_d, i_b].cdofd_vel = (
#                             gu.motion_cross_motion(
#                                 cvel_ang,
#                                 cvel_vel,
#                                 self.dofs_state[i_d, i_b].cdof_ang,
#                                 self.dofs_state[i_d, i_b].cdof_vel,
#                             )
#                         )
#                     for i_d in range(dof_start, j_info.dof_end):
#                         cvel_vel = cvel_vel + self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].vel
#                         cvel_ang = cvel_ang + self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].vel

#             self.links_state[i_l, i_b].cd_vel = cvel_vel
#             self.links_state[i_l, i_b].cd_ang = cvel_ang
#             self.links_state[i_l, i_b].vel = cvel_vel
#             self.links_state[i_l, i_b].ang = cvel_ang

nd1i = ti.ndarray(int, shape=(10))
nd1 = ti.ndarray(float, shape=(10))
nd2 = ti.ndarray(float, shape=(10, 7))
rigid_solver = gs.engine.solvers.RigidSolver()
print('')

print("dir(foo)", dir(foo), type(foo), foo)
foo.background_compile()
print("after background compile foo")
print('')

foo()
print("after call foo")
print('')

bar()
print("after call bar")
print('')

rigid_solver._kernel_init_joint_fields(nd1i, nd2, nd1i, nd1i, nd1i, nd1i, nd2)
rigid_solver._kernel_forward_kinematics_links_geoms(nd1i)
