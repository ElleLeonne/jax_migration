import jax
import jax.numpy as jnp
import flax.linen as nn
from modeling_emme.emme_model import EmmeLMModel
from utils.preprocess import load_config

def count_params(params):
    """jnp lacks .prod utility. Clunky, but it works in a pinch."""
    import numpy as np
    total_params = 0
    for p in jax.tree_leaves(params):
        total_params += np.prod(p.shape)
    return total_params

jax.config.update('jax_platform_name', 'cpu') #Jax model won't init on 16gb vram natively. This tells it to try on CPU
config = load_config("./modeling_emme/config.yml") #This function returns a config object, wrapper not necessary
ckpt_dir = 'emme_tmp'
emme_class = EmmeLMModel(config)

class MigrateModel():
    def __init__(self):
        self.init = nn.initializers.variance_scaling(scale=1e-7, mode='fan_in', distribution='truncated_normal')
        self.rng = jax.random.PRNGKey(0)

    def load_flax_model(self, model, init=True):
        """Set init=False to return a lazily instantiated model"""
        dummy = jnp.ones((1, 2), dtype=jnp.float16)  # Adjust shape as needed
        dummy2 = jnp.ones((1, 2), dtype=jnp.int16) #Some spare dummy data arrays, if needed
        dummy3 = jnp.expand_dims(dummy, (1, -1))
        rng = jax.random.PRNGKey(0)
        if init is True:
            return model.init(rng, dummy2)
        else:
            return jax.eval_shape(model.init, rng, dummy2)

    def print_pt_layer_names(self, pt_model):
        """Prints the name of all Pytorch layers, in the order that pt_strip_layers will return their weights"""
        for name, param in pt_model.named_parameters():
            if param.requires_grad:
                print(name)

    def pt_pull_layers(self, pt_model):
        self.pt_layers = []
        for name, param in pt_model.named_parameters():
            if param.requires_grad:
                self.pt_layers.append(param.data)
        self.total = len(self.pt_layers)
        self.current = 0

    def to_jnp(self, kernel):
        """Flips the position of pytorch weights to match Flax's layer paradigm"""
        kernel = jnp.array(kernel)
        return jnp.transpose(kernel, (1, 0))
    
    def generate_fill(self, size):
        """Generates additional parameters, if needed"""
        return self.init(self.rng, (size, size))
    
    def next_layer(self):
        self.current += 1
        return self.pt_layers.pop(0)

    def __call__(self, flax_model, pt_model):
        self.pt_pull_layers(pt_model) #self.pt_layers is now populated
        del pt_model #Free up some memory
        for i in range(22): #Body layers have no abstraction layers between them
            
            #attn
            flax_model["params"][f"layer_{i}"]["attention"]["q_proj"]["kernel"] = self.to_jnp(self.next_layer())
            flax_model["params"][f"layer_{i}"]["attention"]["k_proj"]["kernel"] = self.to_jnp(self.next_layer())
            flax_model["params"][f"layer_{i}"]["attention"]["v_proj"]["kernel"] = self.to_jnp(self.next_layer())
            flax_model["params"][f"layer_{i}"]["attention"]["o_proj"]["kernel"] = self.to_jnp(self.next_layer())
            
            #mlp
            flax_model["params"][f"layer_{i}"]["decoder_mlp"]["mlp_gate"]["kernel"] = self.to_jnp(self.next_layer())
            flax_model["params"][f"layer_{i}"]["decoder_mlp"]["mlp_out"]["kernel"] = self.to_jnp(self.next_layer())
            flax_model["params"][f"layer_{i}"]["decoder_mlp"]["mlp_in"]["kernel"] = self.to_jnp(self.next_layer())

            #layer norms
            flax_model["params"][f"layer_{i}"]["input_norm"]["weight"] = jnp.array(self.next_layer())
            flax_model["params"][f"layer_{i}"]["output_norm"]["weight"] = jnp.array(self.next_layer())

        if self.current == self.total: #check sum
            return flax_model
        else:
            raise Exception

migrator = MigrateModel()
pt_model = your_pytorch_model
flax_model = migrator.load_flax_model(emme_class)
flax_model = flax_model.unfreeze()
flax_model = migrator(flax_model, pt_model)
del pt_model
print(flax_model)
