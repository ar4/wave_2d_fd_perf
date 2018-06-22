import torch
from wave_2d_fd_perf.propagators import VPytorch

class VPytorch1(VPytorch):

    def __init__(self, model, dx, dt=None, align=None):
        pad = 0
        super(VPytorch1, self).__init__(model, pad, dx, dt, align)

    def step(self, nt, source_amplitude, sources_x, sources_y):

        source_amplitude = torch.tensor(source_amplitude)
        sources_x = torch.tensor(sources_x).long()
        sources_y = torch.tensor(sources_y).long()

        for i in range(nt):
            lap = torch.nn.functional.conv2d(self.wfc[..., :self.nx],
                                             self.kernel2d,
                                             padding=(8, 8))
            self.wfp[0, 0, :, :self.nx] = \
                    (self.model[0, 0, :, :self.nx] * lap[0, 0]
                     + 2 * self.wfc[0, 0, :, :self.nx]
                     - self.wfp[0, 0, :, :self.nx])

            self.wfp[0, 0, sources_y, sources_x] += \
                    (source_amplitude[:, i]
                     * self.model[0, 0, sources_y, sources_x])

            self.wfc, self.wfp = self.wfp, self.wfc

        return self.wfc[0, 0, :, :self.nx].numpy()
