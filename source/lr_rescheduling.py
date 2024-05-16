import torch
import torch.nn as nn
import torch.optim as optim

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#option 1
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1, patience = 2,
                                                       threshold=1e-4, threshold_mode="rel")

#option 2
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# factor = factor by which the lr will be decreased - default 0.1
# patience = n_epochs with no improvement after which the lr will be decreased - default 10
# threshold - default 1e-4
# threshold_mode - default rel