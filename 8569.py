import taichi as ti


@ti.kernel
def WriteSingleInt(field: ti.template()):
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
        for _ in range(loopCount):
            fields = []
            for _ in range(batchSize):
                new_field = CreateField()
                print('new field ', new_field.snode.snode_tree_id)
                fields += [new_field]
            for field in fields:
                print('field before ', field.snode.snode_tree_id)
                WriteSingleInt(field)
                print('   ... after ', field.snode.snode_tree_id)

            allFields += fields

    # alloc and write 'size' to a field            
    intFieldA = CreateField()
    print('intFieldA ', intFieldA.snode.snode_tree_id)
    # print(intFieldA.snode)
    # print(intFieldA.snode.parent()) 
    # print('intFieldA', intFieldA[None])
    WriteSingleInt(intFieldA)
    print('intFieldA ', intFieldA.snode.snode_tree_id)
    # print('intFieldA', intFieldA[None])

    # alloc another field after that
    intFieldB = CreateField()
    print('intFieldB ', intFieldB.snode.snode_tree_id)

    # use 'size' from earlier field to create another new field
    print('intFieldA ', intFieldA.snode.snode_tree_id)
    unusedIntFieldC = CreateField(intFieldA[None])
    print('intFieldA ', intFieldA.snode.snode_tree_id)
    print('unusedIntFieldC ', unusedIntFieldC.snode.snode_tree_id)

    # crash:
    # print(intFieldB.snode)
    # print(intFieldB.snode.parent()) 
    print("crash here:")
    print('intFieldA ', intFieldA.snode.snode_tree_id)
    print('intfield ', intFieldB.snode.snode_tree_id)
    WriteSingleInt(intFieldB)
    print("did not crash ?!?")


if __name__ == "__main__":
    main()