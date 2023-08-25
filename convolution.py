from __future__ import absolute_import

import tensorflow as tf
import numpy as np

def padding_number(filter_size, field_size, stride):
	# when strip = 1, p = filter_size - 1
    p = stride * field_size + filter_size - field_size - stride
    if p % 2 == 0:
        # If n is even, split it into two equal halves
        return p // 2, p // 2
    else:
        # If n is odd, split it into two integers as close as possible
        return p // 2, p // 2 + 1

def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    num_examples = inputs.shape[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    in_channels = inputs.shape[3]

    filter_height = filters.shape[0]
    filter_width = filters.shape[1]
    filter_size = int(filter_height * filter_width)

    filter_in_channels = filters.shape[2]
    filter_out_channels = filters.shape[3]

    examples_stride = strides[0]
    stride_height = strides[1]
    stride_width = strides[2]
    channels_stride = strides[3]

    assert(filter_in_channels == in_channels)
    assert(filter_width <= in_width)
    assert(filter_height <= in_height)

    # 0. Calculate padding 
    p_inputs = inputs
    if padding == "SAME":
        height_pad_top, height_bottom = padding_number(filter_height,
                                                       in_height, stride_height)
        weight_pad_left, weight_right = padding_number(filter_width,
                                                        in_width, stride_width)
        # define paddings for each dimension
        paddings = tf.constant([[0, 0],   # No padding for batch dimension
                                [height_pad_top, height_bottom], # height dimension
                                [weight_pad_left, weight_right], # width dimension
                                [0, 0]])  # No padding for channel dimension 
        p_inputs = tf.pad(inputs, paddings, "CONSTANT")

    # Calculate output dimensions
    output_height = int((p_inputs.shape[1] - filter_height) / stride_height + 1)
    output_width = int((p_inputs.shape[2]- filter_width) / stride_width + 1)

    # 1. Filters, convert to 
    # shape(in_channels, filter_size, output_channels)
    # from [filter_height, filter_width, in_channels, out_channels]
    tp_filters = tf.transpose(filters, perm=[2, 0, 1, 3])
    new_filters = tf.reshape(tp_filters,
                                 shape = [-1, filter_size, filter_out_channels])

    # 2. convert p_inputs to
    # shape(num_examples, in_channels, output_height, output_width, filter_size)

    # 2.1 from shape [num_examples, in_height, in_width, in_channels] to
    # shape(num_examples, in_channels, in_height, in_width)
    tp_inputs = tf.transpose(p_inputs, perm=[0, 3, 1, 2])

    # 2.2 indices
    # with shape(in_channels, output_height, output_width, filter_size, 2)
    indices = np.zeros((in_channels,
                        output_height,
                        output_width,
                        filter_size, 2), dtype = np.int32)
    for h in range(0, output_height):
        y0 = stride_height * h
        for w in range(0, output_width):
            x0 = stride_width * w
            for fh in range(0, filter_height):
                yh = y0 + fh
                for fw in range(0, filter_width):
                    xh = x0 + fw
                    cs = np.arange(0, in_channels, channels_stride)
                    indices[cs, h, w, filter_width * fh + fw] = [yh, xh]
    # 2.3. transfer inputs to
    # shape(num_examples, in_channels, output_height, output_width, filter_size)
    new_inputs = tf.stack(
        [tf.gather_nd(tp_input, indices, batch_dims=1)
         for tp_input in tp_inputs[::examples_stride]])
    
    new_inputs = tf.reshape(new_inputs, (num_examples, in_channels,
                            output_height, output_width, filter_size))

    # 3. matrix multiple, got
    outputs = []
    for idx, new_input in enumerate(new_inputs):
        out = np.zeros((output_height, output_width, filter_out_channels),
                        dtype = np.float32)
        for c in range(0, in_channels, channels_stride):
            out += tf.matmul(new_input[c], new_filters[c])
        outputs.append(out)

    # 4. shape(num_examples, output_height, output_width, output_channels])
    return tf.stack(outputs)
    
