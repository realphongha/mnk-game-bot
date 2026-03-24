import time
import asyncio
import math
import uuid
import logging
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ResidualBlock(nn.Module):
    def __init__(self, channels, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ResidualNet(nn.Module):
    def __init__(self, m, n, k, in_channels=2, conv_channels=16, res_blocks=1):
        super(ResidualNet, self).__init__()
        # handle near-edge cases
        # self.padding = math.ceil(k / 2)
        self.padding = 1
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        if res_blocks:
            self.res = nn.Sequential(*[
                ResidualBlock(conv_channels, self.padding) for i in range(res_blocks)
            ])
        else:
            self.res = nn.Identity()
        self.policy = nn.Sequential(
            nn.Conv2d(conv_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2 * m * n, m * n),
        )

        self.value = nn.Sequential(
            nn.Conv2d(conv_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(m * n, 1),
            nn.Tanh()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.res(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


class BatchInferResidualNet:
    def __init__(self, m, n, k, in_channels=2, conv_channels=16, res_blocks=1,
                 batch_size=8, batch_timeout=1.0):
        self.net = ResidualNet(m, n, k, in_channels, conv_channels, res_blocks)
        self.futures = {}
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.queue = asyncio.Queue(maxsize=batch_size * 2)
        self.loop = None
        self.loop_thread = None

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        self.consumer_task = asyncio.run_coroutine_threadsafe(
            self.run(), self.loop)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def stop(self):
        # 5. Shut down the asyncio loop and thread
        if self.loop:
            self.consumer_task.cancel()
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread:
            self.loop_thread.join()

    def __del__(self):
        self.consumer_task.cancel()

    async def call(self, x):
        req_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.futures[req_id] = future
        await self.queue.put((req_id, x))
        res = await future
        del self.futures[req_id]
        return res

    def __call__(self, x):
        if not self.loop:
            raise RuntimeError("Must call start() before calling __call__()")
        future = asyncio.run_coroutine_threadsafe(self.call(x), self.loop)
        policy, value = future.result()
        return policy, value

    def _infer(self, batch_tensor):
        with torch.inference_mode():
            policy, value = self.net(batch_tensor)
            # Detach from GPU and move to CPU immediately
            # so the asyncio thread doesn't hold GPU resources.
            return policy.cpu(), value.cpu()

    async def run(self):
        while True:
            # Wait for the *first* request to arrive
            first_request_id, first_request_data = await self.queue.get()
            requests = [(first_request_id, first_request_data)]
            request_ids = [first_request_id]
            batch_data = [first_request_data.squeeze(0)]

            # start batching
            start = time.monotonic()
            while len(requests) < self.batch_size:
                remaining = self.batch_timeout - (time.monotonic() - start)
                if remaining <= 0:
                    break
                try:
                    req_id, req_data = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining)
                    requests.append((req_id, req_data))
                    request_ids.append(req_id)
                    batch_data.append(req_data.squeeze(0))
                except asyncio.TimeoutError:
                    # No more items? Break and run the batch.
                    break

            print(f"Processing batch of size {len(requests)}...")
            batch_tensor = torch.stack(batch_data, dim=0).float()
            # Run the blocking model inference in a separate thread
            # so we don't block the main asyncio loop!
            policy, value = await asyncio.to_thread(self._infer, batch_tensor)

            for (req_id, _), p, v in zip(requests, policy, value):
                future = self.futures.get(req_id)
                if future and not future.done():
                    future.set_result((p, v))


MODELS = {
    "residual": ResidualNet,
    "batch_infer_residual": BatchInferResidualNet,
}


if __name__ == "__main__":
    # import numpy as np
    # from board_state.board import to_board, to_bitboard
    #
    # board = [[[1, 0, 1, 0],
    #           [0, 1, 0, 1],
    #           [1, 0, 0, 0],
    #           [0, 1, 0, 0]],
    #          [[0, 1, 0, 0],
    #           [0, 0, 0, 0],
    #           [0, 1, 0, 0],
    #           [1, 0, 0, 0]]]
    # board = np.array(board).astype(np.float32)
    # m = board.shape[1]
    # n = board.shape[2]
    # k = 3
    # inp = torch.tensor(board).float().unsqueeze(0)
    # net = ResidualNet(m, n, 3)
    # print(net)
    # policy, value = net(inp)
    # print(policy)
    # print(value)

    def demo():
        batch_net = BatchInferResidualNet(7, 7, 5, batch_size=8, batch_timeout=1.0)
        batch_net.start()

        def worker(idx):
            x = torch.randn(1, 2, 7, 7)
            p, v = batch_net(x)
            print(f"thread {idx}: policy {tuple(p.shape)} value {tuple(v.shape)}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        time.sleep(1)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        time.sleep(1)
        batch_net(torch.randn(1, 2, 7, 7))

    demo()
