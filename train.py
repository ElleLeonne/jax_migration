import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import sentencepiece as spm
from optax import adamw

from model import LMHead
from utils.preprocess import load_config, Preprocessor, WrapDict
from utils.trainer import train_step, eval_step
from utils.save_load import Checkpoints

"""Basic training loop"""

train_config = WrapDict({ #WrapDict points class attributes to dict keys, so that we can access epochs with train_config.epoch
    "epochs": 10,
    "learning_rate": 3e-4,
    "eval_freq": 1, #Eval every n epochs
})

#Each sample in data requires these things, if using our flax model
data = [("input_ids", "attn_mask", "position_ids", "labels")] 
eval_data = ["input_ids"]

def train(state, train_config):

    for epoch in range(train_config.epochs):
        for sample in data:
            state, metrics = train_step(state, sample)

        if epoch % train_config.eval_freq== 0:
            eval_metrics = []
            for batch in eval_data:
                metrics = eval_step(state, batch)
                eval_metrics.append(metrics)
            eval_metrics = jax.device_get(eval_metrics)
            eval_metrics = {k: jnp.mean(v) for k, v in eval_metrics.items()}
            print(f"Epoch {epoch}: {eval_metrics}")

    checks.save(params)

if __name__ == "__main__":
    config = load_config("./modeling_emme/config.yml")
    checks = Checkpoints(config)
    model = LMHead(config)
    tokenizer = spm.SentencePieceProcessor("./tokenizer/tokenizer.model") #Tokenizer not included
    processor = Preprocessor(config, tokenizer)
    params = model.init(jax.random.PRNGKey(0), data[0])
    
    train(train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=adamw(train_config.learning_rate)
        ), train_config)
