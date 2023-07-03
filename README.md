A dumping ground for the various Jax utilities needed to continue my work.
It's worth noting that many of these tools are tailor made for my specific use case.
Not only may they require some adjustments, but posting them here often requires removal of proprietary data,
which may have the unintended consequence of stripping away ease of use or showing off some more advanced functionality.

Jax is a low-level library, and is often better suited to custom built tools over generalizable libraries. Still, I hope this may still be useful to someone.

#Table of Contents

Migrator: A tool to convert Pytorch weights to Jax Pytrees. Strips off PT weights into a list, and leaves you to assign the leaves yourself.

Preprocessor: A class for preparing attention masks, position_ids, and labels. Pads, caches, and prepares data.

Checkpointer: A tool for saving model shards in Orbax. Built to handle low memory systems where weights may cause overflow issues.

Train & utils.trainer: Training pipeline tools for basic implementation of the flax_model. Designed to work with Preprocessor outputs.
