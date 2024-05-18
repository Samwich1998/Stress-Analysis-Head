import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


class complexHelperMethods:

    @staticmethod
    def applyComplexTransformation(complexData, convolutionLayer):
        # Unpack the real and imaginary components
        realData = complexData.real
        imagData = complexData.imag

        # Convolve the data.
        realData = convolutionLayer(realData)
        imagData = convolutionLayer(imagData)

        # Unpack the real and imaginary components.
        complexData = torch.complex(realData, imagData)

        return complexData

    @staticmethod
    def apply_phase_shift(fourier_data, phaseShifts):
        # Calculate the angles and magnitudes of the Fourier data.
        angles = fourier_data.cdouble().angle() + phaseShifts  # Broadcasting phase shifts
        magnitudes = fourier_data.cdouble().abs()

        # Apply phase shifts
        adjusted_fourier = magnitudes * (1j * angles.cdouble()).exp()
        return adjusted_fourier.cfloat()

    @staticmethod
    def complexInterp(complexData, target_size, useInterpolate=True):
        # Unpack the real and imaginary components
        realData = complexData.real
        imagData = complexData.imag

        if useInterpolate:
            # Interpolate using 1D interpolation
            realInterp = F.interpolate(realData, size=target_size, mode='linear', align_corners=False, antialias=False)
            imagInterp = F.interpolate(imagData, size=target_size, mode='linear', align_corners=False, antialias=False)
        else:
            # Interpolate using 1D interpolation
            interpolationModel = torch.nn.Upsample(size=target_size, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None)
            realInterp = interpolationModel(realData)
            imagInterp = interpolationModel(imagData)

        # Recombine the real and imaginary components
        interpolated = torch.complex(realInterp, imagInterp)

        return interpolated

    @staticmethod
    def complexInterp2(complexData, target_size, useInterpolate=True):
        # Ensure target_size is a tuple (needed for F.interpolate)
        if isinstance(target_size, int):
            target_size = (target_size,)

        # Unpack the real and imaginary components
        realData = complexData.real
        imagData = complexData.imag

        # Expand dimensions to [batch, channels, height, width]
        realData = realData.unsqueeze(2)  # Add height dimension
        imagData = imagData.unsqueeze(2)  # Add height dimension

        if useInterpolate:
            # Interpolate using 1D interpolation
            realInterp = F.interpolate(realData, size=(1, target_size[0]), mode='bicubic', align_corners=False).squeeze(2)
            imagInterp = F.interpolate(imagData, size=(1, target_size[0]), mode='bicubic', align_corners=False).squeeze(2)
        else:
            # Interpolate using 1D interpolation
            interpolationModel = torch.nn.Upsample(size=(1, target_size[0]), scale_factor=None, mode='bicubic', align_corners=None, recompute_scale_factor=None)
            realInterp = interpolationModel(realData)
            imagInterp = interpolationModel(imagData)

        # Recombine the real and imaginary components
        interpolated = torch.complex(realInterp, imagInterp)

        return interpolated
