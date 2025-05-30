class TransformTrackResult(object):
    def __init__(self, position, template_size, confidence, scale, rotation, distance, is_reasonable_distance, mask=None):
        self.position = position
        self.template_size = template_size
        self.confidence = confidence
        self.scale = scale
        self.rotation = rotation
        self.mask = mask
        self.distance = distance
        self.is_reasonable_distance = is_reasonable_distance

    def __repr__(self):
        return f"""{{
            {self.position},
            {self.confidence},
            {self.distance},
            {self.is_reasonable_distance},
        }}"""