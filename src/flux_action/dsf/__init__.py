"""Depth Self-Flow (DSF) fine-tuning: one network, two noise levels.

Around gripper events the robot and the handled object are hidden in ONE exterior camera with a high noise level
``t_high`` (the latent tokens they touch and the actions of those latent frames); everything else gets ``t_low``.
The other exterior camera stays at the low level as the reference, so to redraw the hidden camera the model has to
work out depth and the robot's position from the other view. The loss is the plain flow-matching loss.

The masks come from ``dsf_flux`` (one ``blocks_<run>.npz`` per Cosmos3-DROID episode: the hidden pixels of every
high block, per exterior camera). ``masks`` turns a window's pixel masks into token masks after the window's own crop
augmentation; ``dataset`` is the trainer's dataset plug; ``noise`` draws the two levels.
"""
