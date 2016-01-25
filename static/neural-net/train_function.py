# From Keras: https://github.com/fchollet/keras

import tensorflow as tf

class Function(object):

    def __init__(self, inputs, outputs, updates=[], session=None):
        assert type(inputs) in {list, tuple}
        assert type(outputs) in {list, tuple}
        assert type(updates) in {list, tuple}
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.session = session
        with tf.control_dependencies(self.outputs):
            self.updates = [new_val for (val, new_val) in updates]

    def __call__(self, inputs):
        assert type(inputs) in {list, tuple}
        names = [value.name for value in self.inputs]
        feed_dict = dict(zip(names, inputs))
        updated = self.session.run(self.outputs + self.updates, feed_dict=feed_dict)
        return updated[:len(self.outputs)]

def function(inputs, outputs, updates=[], session=None):
    return Function(inputs, outputs, updates=updates, session=session)

