import taichi as ti
ti.init(arch=ti.cpu, offline_cache=False, advanced_optimization=True,
        cpu_max_num_threads=1, debug=True, log_level=ti.DEBUG, print_ir=True)


@ti.dataclass
class DataClassTest:
    v: ti.types.vector(3, dtype=ti.f64)

    @ti.func
    def manipulate_elements(self) -> float:
        self.v = [1.23, 2.34, 3.45]
        idx = 1
        return self.v[idx] # crash
        # return self.v[1]  # work as expected

d = DataClassTest.field(shape=())

@ti.kernel
def my_kernel() -> float:
    val = d[None].manipulate_elements()
    return val

def main():
    print('1')
    ret = my_kernel()
    print('ret', ret)
    print('2')

main()

"""
kernel {
  $16 = alloca @tmp0
  <*f32>@tmp0 = 1
  $18 = alloca @tmp1
  <*f32>@tmp1 = 2
  $20 = alloca @tmp2
  <*f32>@tmp2 = 3
  <*[Tensor (3) f64]>[#@tmp0 (snode=S2place<f64>), #@tmp4 (snode=S3place<f64>), #@tmp8 (snode=S4place<f64>)] (3, dynamic_index_stride = 8)[] = [<*f32>@tmp0, <*f32>@tmp1, <*f32>@tmp2] (dt=[Tensor (3) f32])
  $23 = alloca @tmp3
  <*i32>@tmp3 = 1
  $25 = alloca @tmp4
  <*i32>@tmp4 = <*i32>@tmp3
  $27 = alloca @tmp5
  <*f32>@tmp5 = (cast_value<f32> <*f64><*[Tensor (3) f64]>[#@tmp0 (snode=S2place<f64>), #@tmp4 (snode=S3place<f64>), #@tmp8 (snode=S4place<f64>)] (3, dynamic_index_stride = 8)[][<*i32>@tmp4])
  $29 = alloca @tmp6
  <*f32>@tmp6 = <*f32>@tmp5
  $31 : return [(cast_value<f32> <*f32>@tmp6)]
}


********************************* After Simplified II
kernel {
  <u1> $84 = const true
  85 : assert $84, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of size () with indices ()
"
  <*[Tensor (3) f64]> $71 = global ptr [S2place<f64>], index [] activate=true
  <f64> $76 = const 1
  <f64> $77 = const 2
  <f64> $78 = const 3
  <[Tensor (3) f64]> $79 = [$76, $77, $78]
  $46 : global store [$71 <- $79]
  <i32> $82 = const 8
  <u1> $87 = const true
  88 : assert $87, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of size () with indices ()
"
  <*f64> $74 = global ptr [S2place<f64>], index [] activate=false
  <*f64> $75 = shift ptr [$74 + $82]
  <f64> $61 = global load $75
  <f32> $62 = cast_value<f32> $61
  $69 : return tmp62
}


********************************* before cfg
kernel {
  $0 = offloaded
  body {
    <u1> $1 = const true
    2 : assert $1, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of size () with indices ()
"
    <*[Tensor (3) f64]> $3 = global ptr [S2place<f64>], index [] activate=true
    <f64> $4 = const 1.2300000190734863
    <f64> $5 = const 2.3399999141693115
    <f64> $6 = const 3.450000047683716
    <[Tensor (3) f64]> $7 = [$4, $5, $6]
    $8 : global store [$3 <- $7]
    <i32> $9 = const 8
    <u1> $10 = const true
    11 : assert $10, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of size () with indices ()
"
    <*f64> $12 = global ptr [S2place<f64>], index [] activate=false
    <*f64> $13 = shift ptr [$12 + $9]
    <f64> $14 = global load $13
    <f32> $15 = cast_value<f32> $14
    $16 : return tmp15
  }
}

********************************* after cfg
kernel {
  $0 = offloaded
  body {
    <u1> $1 = const true
    2 : assert $1, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of size () with indices ()
"
    <*[Tensor (3) f64]> $3 = global ptr [S2place<f64>], index [] activate=true
    <f64> $4 = const 1.2300000190734863
    <f64> $5 = const 2.3399999141693115
    <f64> $6 = const 3.450000047683716
    <[Tensor (3) f64]> $7 = [$4, $5, $6]
    $8 : global store [$3 <- $7]
    <u1> $10 = const true
    11 : assert $10, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of size () with indices ()
"
    <f32> $15 = cast_value<f32> $1
    $16 : return tmp15
  }
}

"""
