A basic script for converting Pytorch model weights to Flax.

Some things to note:
Flax weights are stored in PyTree's, a nested dictionary. This will strip the pytorch weights out into a list, and leave you to manually assign them back into the dictionary.
Some proprietary information was stripped out of the model details, or changed into more traditional configurations, so this will likely require some editing to make it work for your needs.
