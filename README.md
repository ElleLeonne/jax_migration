A dumping ground for the various Jax utilities needed to continue my work.
It's worth noting that many of these tools are tailor made for my specific use case.
Not only may they require somea adjustments, but posting them here often requires removal of proprietary data,
which may have the unintended consequence of stripping away ease of use or showing off some more advanced functionality.

Migrator: A tool for converter Pytorch weights to Jax Pytrees. Strips off PT weights into a list, and leaves you to assign the leaves.

Checkpointer: A tool for saving model shards in Orbax.
