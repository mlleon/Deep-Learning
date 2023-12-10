from torch.utils.tensorboard import SummaryWriter
import numpy as np
log_dir="./train_log/test_log_dir"
writer = SummaryWriter(log_dir=log_dir)
for x in range(1, 101) :
    writer.add_scalar('y = 2x', x, 2 * x)
writer.close()