import torch
from wave_2d_fd_perf.propagators import VPytorch

class VPytorch2(VPytorch):

    def __init__(self, model, dx, dt=None, align=None):
        pad = 8
        super(VPytorch2, self).__init__(model, pad, dx, dt, align)
        self.model = self.model[0, 0]
        self.wfc = self.wfc[0, 0]
        self.wfp = self.wfp[0, 0]
        self.lap = torch.zeros_like(self.wfc)

    def step(self, nt, source_amplitude, sources_x, sources_y):

        source_amplitude = torch.tensor(source_amplitude)
        sources_x = torch.tensor(sources_x).long()
        sources_y = torch.tensor(sources_y).long()

        for i in range(nt):
            self.lap[8:-8, 8:self.nx+8] = \
                    (self.fd_coeff[0] *
                     self.wfc[8:-8, 8:self.nx+8] +
                     self.fd_coeff[1] *
                     (self.wfc[8:-8, 9:self.nx+9] +
                      self.wfc[8:-8, 7:self.nx+7] +
                      self.wfc[9:-7, 8:self.nx+8] +
                      self.wfc[7:-9, 8:self.nx+8]) +
                     self.fd_coeff[2] *
                     (self.wfc[8:-8, 10:self.nx+10] +
                      self.wfc[8:-8, 6:self.nx+6] +
                      self.wfc[10:-6, 8:self.nx+8] +
                      self.wfc[6:-10, 8:self.nx+8]) +
                     self.fd_coeff[3] *
                     (self.wfc[8:-8, 11:self.nx+11] +
                      self.wfc[8:-8, 5:self.nx+5] +
                      self.wfc[11:-5, 8:self.nx+8] +
                      self.wfc[5:-11, 8:self.nx+8]) +
                     self.fd_coeff[4] *
                     (self.wfc[8:-8, 12:self.nx+12] +
                      self.wfc[8:-8, 4:self.nx+4] +
                      self.wfc[12:-4, 8:self.nx+8] +
                      self.wfc[4:-12, 8:self.nx+8]) +
                     self.fd_coeff[5] *
                     (self.wfc[8:-8, 13:self.nx+13] +
                      self.wfc[8:-8, 3:self.nx+3] +
                      self.wfc[13:-3, 8:self.nx+8] +
                      self.wfc[3:-13, 8:self.nx+8]) +
                     self.fd_coeff[6] *
                     (self.wfc[8:-8, 14:self.nx+14] +
                      self.wfc[8:-8, 2:self.nx+2] +
                      self.wfc[14:-2, 8:self.nx+8] +
                      self.wfc[2:-14, 8:self.nx+8]) +
                     self.fd_coeff[7] *
                     (self.wfc[8:-8, 15:self.nx+15] +
                      self.wfc[8:-8, 1:self.nx+1] +
                      self.wfc[15:-1, 8:self.nx+8] +
                      self.wfc[1:-15, 8:self.nx+8]) +
                     self.fd_coeff[8] *
                     (self.wfc[8:-8, 16:self.nx+16] +
                      self.wfc[8:-8, 0:self.nx] +
                      self.wfc[16:, 8:self.nx+8] +
                      self.wfc[0:-16, 8:self.nx+8]))

            self.wfp[8:-8, 8:self.nx+8] = \
                    (self.model[8:-8, 8:self.nx+8]
                     * self.lap[8:-8, 8:self.nx+8]
                     + 2 * self.wfc[8:-8, 8:self.nx+8]
                     - self.wfp[8:-8, 8:self.nx+8])

            self.wfp[sources_y + 8, sources_x + 8] += \
                    (source_amplitude[:, i]
                     * self.model[sources_y + 8, sources_x + 8])

            self.wfc, self.wfp = self.wfp, self.wfc

        return self.wfc[8:-8, 8:self.nx+8].numpy()
