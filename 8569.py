import sys
import taichi as ti


@ti.kernel
def WriteSingleIntB(field: ti.template()):
    field[None] = 1

def CreateField(shape = ()):
    return ti.field(int, shape=shape)


def main():
    ti.init(
        ti.arm64,
        # ti.metal,
        #cpu_max_num_threads=4,
        debug=True,
        # verbose=True,
        # print_ir=True,
        # kernel_profiler=True,
        # random_seed=42,
    )

    # hold on to all fields, just in case
    allFields = []

    # do field allocation & assignment in a pattern
    seq = [(13,1), (1,2), (3, 1), (1,5)]
    for loopCount, batchSize in seq:
        print("+++++++++++++++ outer loop", loopCount)
        for j in range(loopCount):
            print('\\\\\\\\\\\\\\\\\\\\\ inner loop', j)
            fields = []
            for _ in range(batchSize):
                print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
                new_field = CreateField()
                print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
                # print('new field ', new_field.snode.snode_tree_id)
                fields += [new_field]
            for field in fields:
                # print('field before ', field.snode.snode_tree_id)
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                WriteSingleIntB(field)
                # print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
                # print('   ... after ', field.snode.snode_tree_id)

            allFields += fields
        # if loopCount == 1:

    # alloc and write 'size' to a field            
    intFieldA = CreateField()  # will be 18
    # print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
    # print('intFieldA ', intFieldA.snode.snode_tree_id)
    # print(intFieldA.snode)
    # print(intFieldA.snode.parent()) 
    # print('intFieldA', intFieldA[None])
    WriteSingleIntB(intFieldA)
    print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
    # print('intFieldA ', intFieldA.snode.snode_tree_id)
    # print('intFieldA', intFieldA[None])

    # alloc another field after that
    intFieldB = CreateField()   # will be 19
    print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())

    # use 'size' from earlier field to create another new field
    # print('intFieldA 123 ', intFieldA.snode.snode_tree_id)  # 18
    # print('intFieldB ', intFieldB.snode.snode_tree_id)      # 0
    # intFieldA[None]
    # print("after calling intFieldA[None]")
    unusedIntFieldC = CreateField(intFieldA[None])   # will be 20 probably
    # print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
    # print('intFieldA ', intFieldA.snode.snode_tree_id)              # 18
    # print('intFieldB ', intFieldB.snode.snode_tree_id)              # 19
    # print('unusedIntFieldC ', unusedIntFieldC.snode.snode_tree_id)  # 0

    # crash:
    print("crash here:")
    # root contains C, presumably
    print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())
    ti.sync()
    # print('after sync')
    # print('intFieldA ', intFieldA.snode.snode_tree_id)              # 18
    # print('intFieldB ', intFieldB.snode.snode_tree_id)              # 19
    # print('unusedIntFieldC ', unusedIntFieldC.snode.snode_tree_id)  # 20
    # print('**** ti.root.finalized', ti.root.finalized, ti.root._get_children())  # []


    rt = ti.lang.impl.get_runtime()
    print(dir(intFieldA.snode))
    print(intFieldA.snode.ptr)
    print(intFieldA.snode.parent().ptr)
    # print('root 18', rt.prog.get_snode_root(18))
    # print('root 19', rt.prog.get_snode_root(19))
    # print('root 20', rt.prog.get_snode_root(20))

    WriteSingleIntB(intFieldB)  # B=19
    # WriteSingleIntB(intFieldA)  # B=18

    print("did not crash ?!?")


if __name__ == "__main__":
    main()