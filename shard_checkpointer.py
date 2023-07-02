import os
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointHandler, AsyncCheckpointer, CheckpointManagerOptions
from flax.training import orbax_utils

ckpt_dir = "./weights"

class Checkpoints():
    """Basic interface for the Orbax checkpointer."""
    def __init__(self, config, path="./weights"):
        self.layers = config.layers
        self.dir = path
        config = CheckpointManagerOptions(max_to_keep=5, best_mode='max')
        self.handler = AsyncCheckpointer(PyTreeCheckpointHandler("final", 10))
        self.manager = CheckpointManager(self.dir, self.handler, config)
    
    def split_layer_list(self, layers, shards):
        """Splits an uneven integer {layers} into an approximately equal list of num {shards} ints"""
        layers_per_shard = layers // shards
        extra_layers = layers % shards
        return [layers_per_shard + (i < extra_layers) for i in range(shards)]

    def shard(self, model, shard_count: int):
        "Takes a Pytree, returns a sharded dict object split into shard_count pieces"
        shards = {}
        layers = self.split_layer_list(self.layers, shard_count)
        iterator = 0
        for shard in range(len(layers)): #Loop through the remaining layers
            for i in range(body.pop()):
                shards.setdefault(f"shard_{shard}", {})[f"layer_{iterator}"] = model["params"]["Left_Body"].pop(f"layer_{iterator}")
                iterator += 1
        del model
        if iterator == self.body_layers:
            return shards #If check sum matchs, return shard dict
        else:
            raise Exception("Fuck.")
    
    def deshard(self, model_class, shard_dict):
        for shard in list(shard_dict.keys()): #We need to be very careful with memory management here.
            for layer in list(shard_dict[f"{shard}"].keys()): #So we instantiate list objects for our loop.
                model_class["params"]["Left_Body"][f"{layer}"] = shard_dict[f"{shard}"].pop(f"{layer}")
                iterator += 1
        del shard_dict
        if iterator == self.body_layers:
            return model_class
        else:
            raise Exception(f"Deshard layer count did not match. Expected {self.body_layers} layers but got {iterator}.")

    def shard_save(self, shard_dict):
        for shard in list(shard_dict.keys()): #Once again, we make sure to be careful with object classes so we can limit memory resources.
            self.handler.save(os.path.join(self.dir, shard), shard_dict.pop(f"{shard}"))

    def shard_checkpoint(self):
        """Saves a checkpoint. This is asynchronous and more memory efficient than a full save, but slower."""
        return

    def load(self, ckpt_name):
        return self.manager.restore(os.path.join(self.dir, ckpt_name))
