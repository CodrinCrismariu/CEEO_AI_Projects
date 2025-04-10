import math

class SsdAnchorsCalculatorOptions:
    def __init__(self, input_size_width, input_size_height, min_scale, max_scale,
                 num_layers, feature_map_width, feature_map_height,
                 strides, aspect_ratios, anchor_offset_x=0.5, anchor_offset_y=0.5,
                 reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0,
                 fixed_anchor_size=False):
        # Size of input images.
        self.input_size_width = input_size_width
        self.input_size_height = input_size_height
        # Min and max scales for generating anchor boxes on feature maps.
        self.min_scale = min_scale
        self.max_scale = max_scale
        # The offset for the center of anchors (in the scale of a cell).
        self.anchor_offset_x = anchor_offset_x
        self.anchor_offset_y = anchor_offset_y
        # Number of output feature maps to generate anchors on.
        self.num_layers = num_layers
        # Sizes of output feature maps to create anchors. Either feature_map size or
        # stride should be provided.
        self.feature_map_width = feature_map_width
        self.feature_map_height = feature_map_height
        self.feature_map_width_size = len(feature_map_width)
        self.feature_map_height_size = len(feature_map_height)
        # Strides of each output feature map.
        self.strides = strides
        self.strides_size = len(strides)
        # List of different aspect ratios to generate anchors.
        self.aspect_ratios = aspect_ratios
        self.aspect_ratios_size = len(aspect_ratios)
        # Whether to reduce the number of boxes in the lowest layer.
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        # An additional anchor is added with this aspect ratio and a scale interpolated
        # between the scale for the current layer and the next layer.
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio
        # Whether to use fixed anchor size (e.g. 1.0 for both width and height).
        self.fixed_anchor_size = fixed_anchor_size

    def to_string(self):
        return ('input_size_width: {:}\ninput_size_height: {:}\nmin_scale: {:}\nmax_scale: {:}\n'
                'anchor_offset_x: {:}\nanchor_offset_y: {:}\nnum_layers: {:}\nfeature_map_width: {:}\n'
                'feature_map_height: {:}\nstrides: {:}\naspect_ratios: {:}\n'
                'reduce_boxes_in_lowest_layer: {:}\ninterpolated_scale_aspect_ratio: {:}\n'
                'fixed_anchor_size: {:}').format(
                self.input_size_width, self.input_size_height, self.min_scale, self.max_scale,
                self.anchor_offset_x, self.anchor_offset_y, self.num_layers,
                self.feature_map_width, self.feature_map_height, self.strides, self.aspect_ratios,
                self.reduce_boxes_in_lowest_layer, self.interpolated_scale_aspect_ratio,
                self.fixed_anchor_size)

class Anchor:
    def __init__(self, x_center, y_center, h, w):
        self.x_center = x_center
        self.y_center = y_center
        self.h = h
        self.w = w

    def to_string(self):
        return 'x_center: {:}, y_center: {:}, h: {:}, w: {:}'.format(self.x_center, self.y_center, self.h, self.w)

def gen_anchors(options):
    anchors = []
    # Verify the options.
    if options.strides_size != options.num_layers:
        print("strides_size and num_layers must be equal.")
        return []
    layer_id = 0
    while layer_id < options.strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios_list = []
        scales = []
        # For same strides, merge anchors.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < options.strides_size and
               options.strides[last_same_stride_layer] == options.strides[layer_id]):
            scale = options.min_scale + (options.max_scale - options.min_scale) * float(last_same_stride_layer) / (options.strides_size - 1.0)
            if (last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer):
                aspect_ratios_list.append(1.0)
                aspect_ratios_list.append(2.0)
                aspect_ratios_list.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                for aspect_ratio_id in range(options.aspect_ratios_size):
                    aspect_ratios_list.append(options.aspect_ratios[aspect_ratio_id])
                    scales.append(scale)
                if options.interpolated_scale_aspect_ratio > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == options.strides_size - 1 else options.min_scale + (options.max_scale - options.min_scale) * float(last_same_stride_layer + 1) / (options.strides_size - 1.0)
                    scales.append(math.sqrt(scale * scale_next))
                    aspect_ratios_list.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        for i in range(len(aspect_ratios_list)):
            ratio_sqrts = math.sqrt(aspect_ratios_list[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)
        if options.feature_map_height_size > 0:
            feature_map_height = options.feature_map_height[layer_id]
            feature_map_width = options.feature_map_width[layer_id]
        else:
            stride = options.strides[layer_id]
            feature_map_height = math.ceil(float(options.input_size_height) / stride)
            feature_map_width = math.ceil(float(options.input_size_width) / stride)
        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    # Here we support anchor_offset by multiplying with cell index.
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    if options.fixed_anchor_size:
                        w = 1.0
                        h = 1.0
                    else:
                        w = anchor_width[anchor_id]
                        h = anchor_height[anchor_id]
                    new_anchor = Anchor(x_center, y_center, h, w)
                    anchors.append(new_anchor)
        layer_id = last_same_stride_layer
    return anchors
