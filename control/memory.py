import torch


class ReplayBuffer:

    def __init__(self,
                 device,
                 capacity,
                 batch_size,
                 dim_state,
                 dim_action,
                 batch_abs_mode=True,
                 min_samples=1000,
                 update_interval=1
                 ):
        """
        :param device:
        :param capacity:
        :param batch_size:
        :param dim_state:
        :param dim_action:
        :param batch_abs_mode:
                                True => fixed batch size
                                False => batch size is % of items in buffer specified in param batch_size
        :param min_samples:
        :param update_interval: period for update signal,
                                signals once there is new update_interval (or more) items in memory
        """

        self.device = device
        self.batch_size = batch_size
        self.batch_abs_mode = batch_abs_mode
        self.min_samples = min_samples
        self.update_interval = update_interval
        self.capacity = capacity
        self.buffer_idx = 0
        self.push_ctr = 0
        self.initialized_min_memory = False
        self.initialized_buffers = False

        def pt_empty(dim): return torch.empty(size=(capacity, dim), device=device)

        self.state_buffer = pt_empty(dim_state)
        self.action_buffer = pt_empty(dim_action)
        self.reward_buffer = pt_empty(1)
        self.next_state_buffer = pt_empty(dim_state)
        self.done_buffer = pt_empty(1)

    def push(self,
             state,
             action,
             reward,
             next_state,
             done
             ):

        self.state_buffer[self.buffer_idx] = state
        self.action_buffer[self.buffer_idx] = action
        self.reward_buffer[self.buffer_idx] = reward
        self.next_state_buffer[self.buffer_idx] = next_state
        self.done_buffer[self.buffer_idx] = done

        self.buffer_idx += 1
        if self.buffer_idx % self.capacity == 0:
            self.initialized_buffers = True
            self.buffer_idx = 0

        self.push_ctr += 1

        if not self.initialized_min_memory:
            self.initialized_min_memory = self.buffer_idx >= self.min_samples

        return self.update_flag

    def sample(self, device=None):
        """
        :param device: destination device
        :return: training samples on specified device
        """

        if device is None:
            device = self.device

        self.push_ctr = 0
        idx = torch.randint(0, self.current_capacity, (self.current_batch_size,), device=self.device)
        state_batch = self.state_buffer.index_select(0, idx).to(device)
        action_batch = self.action_buffer.index_select(0, idx).to(device)
        reward_batch = self.reward_buffer.index_select(0, idx).to(device)
        next_state_batch = self.next_state_buffer.index_select(0, idx).to(device)
        done_batch = self.done_buffer.index_select(0, idx).to(device)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    @property
    def initialized(self):
        return self.initialized_buffers

    @property
    def current_batch_size(self):
        return self.batch_size if self.batch_abs_mode else int(self.current_capacity * self.batch_size)

    @property
    def current_capacity(self):
        return self.capacity if self.initialized_buffers else self.buffer_idx

    @property
    def update_flag(self):
        """ signals once there is new update_interval (or more) items in memory"""
        return self.push_ctr >= self.update_interval and self.initialized_min_memory

    def __len__(self):
        return self.current_capacity
