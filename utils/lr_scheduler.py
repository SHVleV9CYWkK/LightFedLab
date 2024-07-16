class LossBasedLRScheduler:
    def __init__(self, initial_lr, factor, patience, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_loss = float('inf')
        self.loss_increase_count = 0
        self.current_lr = initial_lr

    def step(self, current_loss):
        if current_loss <= self.min_loss:
            self.min_loss = current_loss
            self.loss_increase_count = 0
        else:
            self.loss_increase_count += 1

        if self.loss_increase_count >= self.patience:
            self.adjust_lr()

    def adjust_lr(self):
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if new_lr < self.current_lr:
            self.current_lr = new_lr
            print(f"Learning rate reduced to {self.current_lr}")
            self.loss_increase_count = 0  # Reset counter after adjustment

    def get_lr(self):
        return self.current_lr
