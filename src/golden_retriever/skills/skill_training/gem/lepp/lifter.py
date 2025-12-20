# 1 x c x 64 x 64 -pad-> 1 x c x 96 x 96 -rot_inv->  180 x c x 96 x 96  -fourier-> c x 2k x 96 x 96 -crop-> c x 2k x 65 x 65


import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F


class Transitor:
    def __init__(
        self, device, n_rotations=180, lmax=36, quotient=True, conditioned=False, c=72
    ):
        self.device = device
        self.conditioned = conditioned
        if not conditioned:
            self.kernel = torch.nn.Parameter(
                data=torch.rand(1, 3, 64, 64, device=self.device), requires_grad=True
            )
        self.n_rotations = n_rotations
        self.lmax = lmax
        self.quotient = quotient
        if self.quotient:
            # c: is the number of discrete classes
            # recon_level is the number of uniformly-distributed samples in [0, 2pi)
            self.c = c // 2
            self.recon_level = self.c * 5 * 2
            # print(self.recon_level)
        else:
            self.c = c
            self.recon_level = self.c * 5

    def to_fourier_kernel(self, x=None, plot=False):
        if not self.conditioned:
            x = self.kernel
        x_shape = x.shape
        x = F.pad(x, (16, 16, 16, 16))

        lift_x = x.repeat(self.n_rotations, 1, 1, 1)
        degrees = torch.from_numpy(
            np.linspace(0.0, 360.0, self.n_rotations, endpoint=False, dtype=np.float32)
        ).to(self.device)
        lift_x = K.geometry.rotate(lift_x, degrees, mode="nearest")
        half_length = x_shape[-1] // 2
        pivot = lift_x.shape[-1] // 2
        l, r = pivot - half_length, pivot + half_length + 1
        lift_x = lift_x[:, :, l:r, l:r]
        # print('================',lift_x.shape)
        lift_x_fourier = fft.rfft(lift_x, dim=0)[: self.lmax]
        # print(lift_x_fourier.shape)
        if self.quotient:
            lift_x_fourier[1::2, :, :, :] = torch.complex(
                torch.zeros(1), torch.zeros(1)
            ).to(self.device)

        plot = plot
        if plot:
            lift_x_fourier_real_img = torch.view_as_real(lift_x_fourier)
            kernel_real = lift_x_fourier_real_img[:, :, :, :, 0]
            kernel_imag = lift_x_fourier_real_img[:, :, :, :, 1]
            # todo plot the kernel
            fig, axs = plt.subplots(2, 8)
            for i in range(8):
                axs[0, i].set_title("frequency {}".format(i), fontsize=16)
                axs[0, i].imshow(kernel_real[i, 0, :, :].cpu().detach().numpy())
                axs[1, i].imshow(kernel_imag[i, 0, :, :].cpu().detach().numpy())
            plt.show()

            for i in range(10):
                S = 6 + 65
                fig, axs = plt.subplots(1, 1, figsize=(24, 24))
                X, Y = np.meshgrid(range(S - 6), range(S - 7, -1, -1))
                # print(X.shape,Y.shape)
                # print(X,Y)
                axs.set_title("frequency {}".format(i))
                axs.quiver(
                    X,
                    Y,
                    kernel_real[i, 0, :, :].cpu().detach().numpy(),
                    kernel_imag[i, 0, :, :].cpu().detach().numpy(),
                    units="xy",
                )
                plt.show()

        return lift_x_fourier

    def ast(self, fourier_kernel, cov_field):
        """
        fourier_kernel: K x C x h x w complex number
        conv_field: 1 x C x H' x W'
        out: 1 x K x H x W complex number
        """
        assert fourier_kernel.shape[1] == cov_field.shape[1]
        kernels = torch.split(fourier_kernel, split_size_or_sections=1, dim=1)
        # print(len(kernels),kernels[0].shape)
        convs = []
        for i, kernel in enumerate(kernels):
            kernel = torch.view_as_real(kernel)
            kernel_real_img = torch.concat(
                (kernel[:, :, :, :, 0], kernel[:, :, :, :, 1]), dim=0
            )
            # print(kernel_real_img.shape)
            # print(conv_field.shape, kernel.shape)
            conv = F.conv2d(
                input=cov_field[:, i, :, :].unsqueeze(dim=1), weight=kernel_real_img
            )
            convs.append(conv)
        convs = torch.cat(convs, dim=0)
        convs = torch.sum(convs, dim=0, keepdim=True)
        # print(convs.shape)
        convs_real = convs[:, : convs.shape[1] // 2, :, :]
        convs_imag = convs[:, convs.shape[1] // 2 :, :, :]
        fft_output = torch.complex(convs_real, convs_imag)
        # print(fft_output.shape)
        return fft_output

    def to_spatial(self, fft_output):
        """
        fft_output: 1 x K x H x W
        output: 1 x C x H x W
        """

        truncated_f = fft_output.shape[1]
        if truncated_f < self.recon_level // 2 + 1:
            fft_output = F.pad(
                fft_output, (0, 0, 0, 0, 0, self.recon_level // 2 + 1 - truncated_f)
            )
        else:
            fft_output = fft_output[:, : self.recon_level // 2 + 1 - truncated_f, :, :]

        output = fft.irfft(
            fft_output,
            dim=1,
        )
        # print(output.shape)
        if self.quotient:
            # the signal has the period of  pi
            # uniform samples are in the range [0, 2*pi)
            output = output[:, : output.shape[1] // 2, :, :]
        output = output.reshape(
            1, self.c, output.shape[1] // self.c, output.shape[-2], output.shape[-1]
        )
        output = torch.max(output, dim=2)[0]
        # print(output.shape)
        return output

        # fourier_kernel = torch.view_as_real(fourier_kernel)
        # print(fourier_kernel.shape)
        # kernels =
        # kernel_real_img = torch.concat((kernel_fourier_even[:, :, :, :, 0], kernel_fourier_even[:, :, :, :, 1]),
        #                                dim=0)


# device = torch.device('cuda')
# lf = Transitor(device=device)
# kernel = lf.to_fourier_kernel()
# conv_field = torch.rand(1,3,320+64,160+64).to(device)
# fft_output = lf.ast(kernel, conv_field)
# print(fft_output.shape)
# output = lf.to_spatial(fft_output)
# print(output.shape)
