""" Please complete your shape inferer below to tell tvm your op's return shape and dtype """

from tvm.relay.custom_op.custom_layer_register import register_shape_inferer


@register_shape_inferer("DetectionOut")
def shape_infer_DetectionOut(types, attrs):
    # please return your result
    # example:
    #     return types[0].shape, types[0].dtype
    # Please delete the code line below to run normally after you complete
    return (1, attrs.keep_top_k, 6), "float32"
