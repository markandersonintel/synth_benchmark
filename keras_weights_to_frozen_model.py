from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.contrib.keras import applications
import tensorflow.contrib.keras as keras
from tensorflow.python.tools.freeze_graph import freeze_graph
import os

def load_and_freeze(model:str="ResNet50", weights_path="weights.h5",
                    model_output_dir="model", frozen_model_name="frozen_model.pb"):
    keras.backend.clear_session()
    keras.backend.set_learning_phase(0)
    model = getattr(applications, model)(weights=None)
    model.load_weights(weights_path)
    print("")
    tf.saved_model.simple_save(keras.backend.get_session(), model_output_dir, inputs={"input": model.inputs[0]},
                               outputs={"output": model.outputs[0]})
    freeze_graph(None,
                          None,
                          None,
                          None,
                          ",".join([op.name.split(":")[0] for op in model.outputs]),
                          None,
                          None,
                          os.path.join(model_output_dir, frozen_model_name),
                          False,
                          "",
                          input_saved_model_dir=model_output_dir)
    print("Saved_model data saved at {}.  Frozen model file saved to {}/{}".format(
        model_output_dir, model_output_dir, frozen_model_name))
