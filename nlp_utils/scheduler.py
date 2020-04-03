class lr_scheduler():
    def __init__(self, lr=0.001, patience=10, threshold=0, min_lr=0.0001, rate=0.1, cooldown=0):
        self.lr = lr
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.rate = rate
        self.max_score = -1
        self.max_score_id = -1
        self.current_call = 0
        self.cool_down_begin_id = -1

    def step(self, performance):
        if self.current_call - self.cool_down_begin_id <= self.cooldown:
            if performance > self.max_score:
                self.max_score = performance
                self.max_score_id = self.current_call
            self.current_call += 1
            return False
        if self.max_score == -1:
            self.max_score = performance
            self.max_score_id = 0
        elif performance > self.max_score + self.threshold:
            self.max_score = performance
            self.max_score_id = self.current_call
        elif self.current_call > self.max_score_id + self.patience and self.lr > self.min_lr:
            self.lr = max([self.min_lr, self.lr * self.rate])
            print('[lr scheduler] patience exhausted, will shrink lr to %f' % (self.lr))
            self.cool_down_begin_id = self.current_call
            self.current_call += 1
            return True
        self.current_call += 1
        return False
        